"""Robustness evaluation under capture conditions the clean dataset does not cover.

Every held-out *test* image is re-rendered under a set of synthetic
conditions (cluttered background, low light, blur, sensor noise, rotation,
position/scale shift, and a combined "webcam-like" condition). The full
pipeline - MediaPipe detection, feature construction, classifier - is then
run end to end, so an image counts as wrong if MediaPipe loses the hand.

Complex backgrounds are produced by segmenting the hand with OpenCV GrabCut
(seeded from the MediaPipe landmarks of the clean image) and compositing it
onto cluttered scenes: random crops of two CC-BY photographs bundled with
scikit-learn, and procedurally drawn clutter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import joblib
import numpy as np
import pandas as pd

from . import config, features, viz
from .landmarks import COORD_COLUMNS, HandLandmarker, imread
from .supervised import MODELS, load_dataset

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# ----------------------------------------------------------------------------------
# Hand segmentation (OpenCV GrabCut seeded by landmarks)
# ----------------------------------------------------------------------------------


def hand_mask(image: np.ndarray, landmarks: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Return a soft alpha mask (float32 in [0, 1]) of the hand and forearm."""
    h, w = image.shape[:2]
    pts = (landmarks[:, :2] * [w, h]).astype(np.float32)
    palm = float(np.linalg.norm(pts[0] - pts[9])) or 0.25 * w

    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    hull = cv2.convexHull(pts.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, cv2.GC_PR_FGD)
    cv2.polylines(mask, [hull], True, cv2.GC_PR_FGD, thickness=max(3, int(0.35 * palm)))

    # Forearm: a band running from the wrist away from the fingers, out of the frame.
    d = pts[0] - pts[9]
    d /= np.linalg.norm(d) + 1e-9
    perp = np.array([-d[1], d[0]])
    wrist = pts[0]
    far = wrist + d * 4 * palm
    poly = np.array([wrist + perp * 0.45 * palm, far + perp * 0.6 * palm,
                     far - perp * 0.6 * palm, wrist - perp * 0.45 * palm], np.int32)
    cv2.fillConvexPoly(mask, poly, cv2.GC_PR_FGD)

    # Bones are certainly foreground.
    for a, b in HAND_CONNECTIONS:
        cv2.line(mask, tuple(map(int, pts[a])), tuple(map(int, pts[b])), cv2.GC_FGD,
                 thickness=max(2, int(0.12 * palm)))

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)

    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, comp, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:
        fg = (comp == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(fg.astype(np.float32), (5, 5), 0)


# ----------------------------------------------------------------------------------
# Cluttered backgrounds
# ----------------------------------------------------------------------------------


class BackgroundBank:
    """Random cluttered 256x256 backgrounds: photo crops and procedural clutter."""

    def __init__(self, seed: int = 0) -> None:
        from sklearn.datasets import load_sample_images

        self.rng = np.random.default_rng(seed)
        self.photos = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in load_sample_images().images]

    def photo(self, size: int) -> np.ndarray:
        img = self.photos[self.rng.integers(len(self.photos))]
        h, w = img.shape[:2]
        s = int(self.rng.integers(min(h, w) // 3, min(h, w)))
        y, x = self.rng.integers(0, h - s + 1), self.rng.integers(0, w - s + 1)
        crop = cv2.resize(img[y : y + s, x : x + s], (size, size), interpolation=cv2.INTER_AREA)
        if self.rng.random() < 0.5:
            crop = cv2.flip(crop, 1)
        return crop

    def clutter(self, size: int) -> np.ndarray:
        rng = self.rng
        c0, c1 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        t = np.linspace(0, 1, size)[:, None, None]
        img = (c0 * (1 - t) + c1 * t).repeat(size, axis=1).astype(np.uint8)
        for _ in range(int(rng.integers(15, 35))):
            color = tuple(int(v) for v in rng.integers(0, 256, 3))
            kind = rng.integers(3)
            p = tuple(int(v) for v in rng.integers(0, size, 2))
            if kind == 0:
                q = tuple(int(v) for v in rng.integers(0, size, 2))
                cv2.rectangle(img, p, q, color, -1 if rng.random() < 0.6 else 2)
            elif kind == 1:
                cv2.circle(img, p, int(rng.integers(5, size // 4)), color, -1 if rng.random() < 0.6 else 2)
            else:
                q = tuple(int(v) for v in rng.integers(0, size, 2))
                cv2.line(img, p, q, color, int(rng.integers(1, 6)))
        noise = rng.normal(0, 8, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def sample(self, size: int = 256) -> np.ndarray:
        return self.photo(size) if self.rng.random() < 0.5 else self.clutter(size)


def composite(image: np.ndarray, alpha: np.ndarray, background: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return (image * a + background * (1 - a)).astype(np.uint8)


# ----------------------------------------------------------------------------------
# Conditions
# ----------------------------------------------------------------------------------


@dataclass
class Context:
    rng: np.random.Generator
    landmarks: np.ndarray  # clean-image landmarks (21, 3), used to seed segmentation
    bank: BackgroundBank


def _rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _affine(scale, angle, dx, dy, w, h):
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (dx, dy)
    return M


def _place(img, ctx: Context, background, scale, angle, dx, dy):
    """Move the segmented hand (rotate / scale / translate) and composite it onto ``background``."""
    h, w = img.shape[:2]
    alpha = hand_mask(img, ctx.landmarks)
    M = _affine(scale, angle, dx, dy, w, h)
    hand = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpAffine(alpha, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return composite(hand, alpha, background)


def _wall(img):
    """Plain background in the image's own wall colour."""
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return np.broadcast_to(np.median(border, axis=0), img.shape).astype(np.uint8)


def _shift_scale(img, ctx: Context):
    """Smaller, off-centre hand on the same plain wall."""
    return _place(img, ctx, _wall(img), float(ctx.rng.uniform(0.55, 0.75)), 0.0, *ctx.rng.uniform(-50, 50, 2))


def _low_light(img):
    dark = 255.0 * (img / 255.0) ** 2.2 * 0.55
    return np.clip(dark, 0, 255).astype(np.uint8)


def _background(img, ctx: Context):
    return composite(img, hand_mask(img, ctx.landmarks), ctx.bank.sample(img.shape[0]))


def _webcam_like(img, ctx: Context):
    """Cut-out hand, rotated, shrunk and moved, on a cluttered scene, slightly out of focus."""
    out = _place(img, ctx, ctx.bank.sample(img.shape[0]), float(ctx.rng.uniform(0.6, 0.8)),
                 float(ctx.rng.uniform(-20, 20)), *ctx.rng.uniform(-40, 40, 2))
    return cv2.GaussianBlur(out, (0, 0), 1.2)


CONDITIONS: dict[str, Callable[[np.ndarray, Context], np.ndarray]] = {
    "clean": lambda img, ctx: img,
    "complex background": _background,
    "low light": lambda img, ctx: _low_light(img),
    "blur": lambda img, ctx: cv2.GaussianBlur(img, (0, 0), 2.5),
    "sensor noise": lambda img, ctx: np.clip(img + ctx.rng.normal(0, 15, img.shape), 0, 255).astype(np.uint8),
    "rotation ±30°": lambda img, ctx: _rotate(img, float(ctx.rng.choice([-1, 1]) * ctx.rng.uniform(15, 30))),
    "shift & scale": _shift_scale,
    "webcam-like (combined)": _webcam_like,
}


def clahe(img: np.ndarray) -> np.ndarray:
    """Contrast-limited histogram equalisation on the lightness channel (used as a preprocessing option)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray) -> np.ndarray:
    """5x5 median filter. MediaPipe's palm detector is very sensitive to per-pixel noise."""
    return cv2.medianBlur(img, 5)


def auto_gamma(img: np.ndarray, target: float = 0.5) -> np.ndarray:
    """Adaptive gamma correction that only ever brightens.

    If the mean brightness is below ``target``, gamma is chosen so the mean moves to
    ``target``; frames that are already bright enough are returned unchanged.
    Implemented as a 256-entry lookup table, so it costs well under a millisecond.
    """
    v = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].mean()) / 255.0
    if v >= target:
        return img
    gamma = np.log(target) / np.log(max(v, 0.01))
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


PREPROCESSING: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "CLAHE": clahe, "auto-gamma": auto_gamma, "denoise": denoise,
}

# (condition, preprocessing) pairs evaluated in addition to every raw condition
PREPROCESSED_RUNS = (
    ("low light", "CLAHE"), ("low light", "auto-gamma"), ("sensor noise", "denoise"),
    ("clean", "auto-gamma"), ("clean", "denoise"),
)


# ----------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def extract_condition(name: str, test: pd.DataFrame, preprocess: str | None = None,
                      force: bool = False) -> pd.DataFrame:
    """Render every test image under one condition, run MediaPipe, and cache the landmarks."""
    tag = _slug(name) + (f"__{_slug(preprocess)}" if preprocess else "")
    preprocess = PREPROCESSING[preprocess] if preprocess else None
    path = config.ROBUSTNESS_CACHE / f"{tag}.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    config.ROBUSTNESS_CACHE.mkdir(parents=True, exist_ok=True)
    bank = BackgroundBank(seed=config.SEED)
    rows = []
    with HandLandmarker(static_image_mode=True) as det:
        for i, r in enumerate(test.itertuples(index=False)):
            rng = np.random.default_rng([config.SEED, i])  # same perturbation for every run
            ctx = Context(rng=rng, landmarks=np.array([getattr(r, c) for c in COORD_COLUMNS]).reshape(21, 3),
                          bank=bank)
            img = imread(config.RAW_DIR / r.instance_id)
            img = CONDITIONS[name](img, ctx)
            if preprocess is not None:
                img = preprocess(img)
            d = det.detect(img)
            coords = d.landmarks.reshape(-1).tolist() if d is not None else [np.nan] * 63
            rows.append([r.instance_id, r.label, d is not None, *coords])
    out = pd.DataFrame(rows, columns=["instance_id", "label", "detected", *COORD_COLUMNS])
    out.to_csv(path, index=False)
    return out


def _load_final_models() -> dict[tuple[str, str], object]:
    store = config.PROCESSED_DIR / "models"
    return {(b["key"], b["feature_set"]): b["model"]
            for b in (joblib.load(p) for p in sorted(store.glob("*.joblib")))}


def run(conditions=tuple(CONDITIONS), preprocessed_runs=PREPROCESSED_RUNS) -> pd.DataFrame:
    config.ensure_dirs()
    df, _ = load_dataset()
    test = df[df["split"] == "test"].reset_index(drop=True)
    models = _load_final_models()
    if not models:
        raise FileNotFoundError("No trained models found - run `asl-gesture train` first.")

    runs = [(c, None) for c in conditions] + list(preprocessed_runs)
    rows = []
    for cond, pre in runs:
        label = cond + (f" + {pre}" if pre else "")
        data = extract_condition(cond, test, preprocess=pre)
        detected = data["detected"].to_numpy(bool)
        lm = data[COORD_COLUMNS].to_numpy(np.float64).reshape(-1, 21, 3)
        y = data["label"].to_numpy()
        for (key, fs), model in models.items():
            pred = np.full(len(y), None, dtype=object)
            if detected.any():
                pred[detected] = model.predict(features.build_features(lm[detected], fs))
            rows.append({"condition": label, "model": MODELS[key].name, "feature_set": fs,
                         "detection_rate": round(detected.mean(), 4),
                         "accuracy": round(float((pred == y).mean()), 4),
                         "accuracy_if_detected": round(float((pred[detected] == y[detected]).mean()), 4)
                         if detected.any() else np.nan})
        print(f"{label:28s} detection={detected.mean():.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(config.TABLE_DIR / "robustness.csv", index=False)
    plot(res)
    return res


def plot(res: pd.DataFrame, model_name: str | None = None) -> None:
    viz.setup()
    if model_name is None:  # model with the best mean accuracy across conditions on normalised features
        model_name = (res[res["feature_set"] != "raw"].groupby("model")["accuracy"].mean().idxmax())
    d = res[res["model"] == model_name]
    seen = [c for c in dict.fromkeys(d["condition"]) if not c.startswith("clean + ")]  # sanity rows: table only
    conds = []  # each "X + preprocessing" bar group right after its "X" group
    for c in (c for c in seen if " + " not in c):
        conds += [c] + [x for x in seen if x.startswith(c + " + ")]
    shown = [fs for fs in ("raw", "norm", "norm_aug") if fs in set(d["feature_set"])]
    series = {fs: [d[(d["condition"] == c) & (d["feature_set"] == fs)]["accuracy"].iloc[0] * 100 for c in conds]
              for fs in shown}
    fig, ax = viz.plt.subplots(figsize=(13, 4.4))
    viz.grouped_bars(ax, [c.replace(" + ", "\n+ ").replace(" (", "\n(").replace("complex ", "complex\n")
                          for c in conds], series,
                     value_fmt="{:.0f}")
    ax.set_ylim(0, 105)
    ax.set_ylabel("End-to-end accuracy (%)")
    ax.set_title(f"Robustness of {model_name} under unseen capture conditions (held-out test images)")
    ax.legend(title="feature set", loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    viz.save(fig, config.FIG_DIR / "robustness.png")


def example_figure(image_paths, landmarks_list, out_path=None) -> None:
    """Show the conditions applied to a few images (use your own photos, not the dataset)."""
    viz.setup()
    names = [c for c in CONDITIONS if c != "clean"]
    bank = BackgroundBank(seed=7)
    n = len(image_paths)
    fig, axes = viz.plt.subplots(n, len(names) + 1, figsize=(2 * (len(names) + 1), 2.15 * n))
    axes = np.atleast_2d(axes)
    for i, (p, lm) in enumerate(zip(image_paths, landmarks_list)):
        img = imread(p)
        ctx = Context(rng=np.random.default_rng(i), landmarks=lm, bank=bank)
        for j, name in enumerate(["clean", *names]):
            out = CONDITIONS[name](img, ctx)
            ax = axes[i, j]
            ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(name, fontsize=8.5, loc="center")
    fig.tight_layout()
    viz.save(fig, out_path or config.FIG_DIR / "robustness_examples.png")
