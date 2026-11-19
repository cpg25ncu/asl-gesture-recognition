"""Real-time recognition from a webcam or a video file.

Pipeline per frame::

    OpenCV capture -> preprocessing (optional CLAHE) -> MediaPipe Hands (tracking mode)
    -> isotropic + normalised landmark features -> classifier -> temporal smoothing -> overlay

Latency is measured for every stage so the cost of each part is visible.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np

from . import config, features
from .landmarks import HandLandmarker
from .robustness import HAND_CONNECTIONS, auto_gamma, clahe, denoise

STAGES = ("preprocess", "landmarks", "classify")


@dataclass
class FramePrediction:
    label: str | None  # smoothed label shown to the user (None = no hand / not confident)
    confidence: float
    raw_label: str | None  # this frame's own prediction
    landmarks: np.ndarray | None
    timings_ms: dict = field(default_factory=dict)


class GestureRecognizer:
    """Stateful per-frame recogniser (keeps a short history for temporal smoothing)."""

    def __init__(
        self,
        model_path: Path = config.DEFAULT_MODEL,
        tracking: bool = True,
        use_clahe: bool = False,
        smoothing: int = 5,
        min_confidence: float = 0.5,
        left_hand: bool = False,
        use_denoise: bool = False,
        use_gamma: bool = False,
    ) -> None:
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_set = bundle["feature_set"]
        self.model_name = bundle.get("model_name", type(self.model).__name__)
        self.classes = [str(c) for c in getattr(self.model, "classes_", config.LABELS)]
        self.detector = HandLandmarker(static_image_mode=not tracking, min_tracking_confidence=0.5)
        self.use_clahe = use_clahe
        self.use_denoise = use_denoise
        self.use_gamma = use_gamma
        self.left_hand = left_hand  # the dataset is right-handed; mirror the frame for left-handers
        self.min_confidence = min_confidence
        self.history: deque[np.ndarray] = deque(maxlen=max(1, smoothing))

    def _proba(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(x), dtype=float)[0]
        pred = str(self.model.predict(x)[0])
        return np.array([1.0 if c == pred else 0.0 for c in self.classes])

    def __call__(self, frame_bgr: np.ndarray) -> FramePrediction:
        t = {}
        t0 = time.perf_counter()
        img = cv2.flip(frame_bgr, 1) if self.left_hand else frame_bgr
        if self.use_denoise:
            img = denoise(img)
        if self.use_gamma:
            img = auto_gamma(img)
        if self.use_clahe:
            img = clahe(img)
        t1 = time.perf_counter()
        det = self.detector.detect(img)
        t2 = time.perf_counter()
        t["preprocess"], t["landmarks"] = (t1 - t0) * 1000, (t2 - t1) * 1000

        if det is None:
            self.history.clear()
            t["classify"] = 0.0
            return FramePrediction(None, 0.0, None, None, t)

        h, w = img.shape[:2]
        lm = det.landmarks
        x = features.build_features(features.to_isotropic(lm, w, h), self.feature_set)
        proba = self._proba(x)
        t["classify"] = (time.perf_counter() - t2) * 1000

        self.history.append(proba)
        mean = np.mean(self.history, axis=0)
        best = int(np.argmax(mean))
        raw = self.classes[int(np.argmax(proba))]
        label = self.classes[best] if mean[best] >= self.min_confidence else None
        if self.left_hand:  # draw on the un-mirrored frame
            lm = lm.copy()
            lm[:, 0] = 1 - lm[:, 0]
        return FramePrediction(label, float(mean[best]), raw, lm, t)

    def close(self) -> None:
        self.detector.close()


# ----------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------

ACCENT = (214, 120, 42)  # BGR of #2a78d6
INK = (11, 11, 11)
WHITE = (252, 252, 251)


def draw(frame: np.ndarray, pred: FramePrediction, fps: float, mirror: bool = False) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    if pred.landmarks is not None:
        pts = (pred.landmarks[:, :2] * [w, h]).astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), WHITE, 3, cv2.LINE_AA)
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), ACCENT, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, tuple(p), 4, WHITE, -1, cv2.LINE_AA)
            cv2.circle(out, tuple(p), 3, ACCENT, -1, cv2.LINE_AA)
    if mirror:
        out = cv2.flip(out, 1)

    # Prediction card (top-left) and FPS (top-right).
    text = pred.label if pred.label else ("?" if pred.landmarks is not None else "no hand")
    big = pred.label is not None
    cv2.rectangle(out, (12, 12), (12 + (150 if big else 170), 12 + (86 if big else 44)), WHITE, -1)
    if big:
        cv2.putText(out, text, (24, 76), cv2.FONT_HERSHEY_DUPLEX, 2.0, ACCENT, 3, cv2.LINE_AA)
        cv2.putText(out, f"{pred.confidence:.0%}", (86, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INK, 2, cv2.LINE_AA)
    else:
        cv2.putText(out, text, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INK, 2, cv2.LINE_AA)
    label = f"{fps:4.1f} FPS"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(out, (w - tw - 26, 12), (w - 12, 24 + th), WHITE, -1)
    cv2.putText(out, label, (w - tw - 19, 18 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, INK, 2, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------------
# Streaming loop
# ----------------------------------------------------------------------------------


def run_stream(
    source: str | int = 0,
    model_path: Path = config.DEFAULT_MODEL,
    tracking: bool = True,
    use_clahe: bool = False,
    use_denoise: bool = False,
    use_gamma: bool = False,
    smoothing: int = 5,
    show: bool = True,
    save: Path | None = None,
    mirror: bool | None = None,
    left_hand: bool = False,
    max_frames: int | None = None,
    camera_size: tuple[int, int] = (640, 480),
) -> dict:
    """Run the recogniser on a camera index or a video path. Press ``q`` / ``Esc`` to quit."""
    is_camera = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
    cap = cv2.VideoCapture(int(source) if is_camera else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {source!r}")
    if is_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_size[1])
    mirror = is_camera if mirror is None else mirror

    rec = GestureRecognizer(model_path, tracking=tracking, use_clahe=use_clahe, smoothing=smoothing,
                            left_hand=left_hand, use_denoise=use_denoise, use_gamma=use_gamma)
    writer = None
    frame_times: list[float] = []
    stage_ms = {s: [] for s in STAGES}
    labels: list[str | None] = []
    n_detected = 0
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames is not None and len(frame_times) >= max_frames):
                break
            t0 = time.perf_counter()
            pred = rec(frame)
            dt = time.perf_counter() - t0
            frame_times.append(dt)
            for s in STAGES:
                stage_ms[s].append(pred.timings_ms.get(s, 0.0))
            labels.append(pred.label)
            n_detected += pred.landmarks is not None
            recent = frame_times[-15:]
            fps = len(recent) / sum(recent)

            if show or save:
                vis = draw(frame, pred, fps, mirror=mirror)
                if save:
                    if writer is None:
                        save.parent.mkdir(parents=True, exist_ok=True)
                        rate = cap.get(cv2.CAP_PROP_FPS) or 30
                        writer = cv2.VideoWriter(str(save), cv2.VideoWriter_fourcc(*"mp4v"), rate,
                                                 (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
                if show:
                    cv2.imshow("ASL gesture recognition  (q to quit)", vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        cap.release()
        rec.close()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    n = len(frame_times)
    if n == 0:
        return {"frames": 0}
    total_ms = np.array(frame_times) * 1000
    return {
        "frames": n,
        "model": rec.model_name,
        "feature_set": rec.feature_set,
        "tracking": tracking,
        "clahe": use_clahe,
        "denoise": use_denoise,
        "auto_gamma": use_gamma,
        "processing_fps": round(1000 / float(np.mean(total_ms)), 1),
        "latency_ms_mean": round(float(np.mean(total_ms)), 2),
        "latency_ms_p95": round(float(np.percentile(total_ms, 95)), 2),
        **{f"{s}_ms_mean": round(float(np.mean(v)), 2) for s, v in stage_ms.items()},
        "hand_detected_rate": round(n_detected / n, 3),
        "labels": labels,
    }


# ----------------------------------------------------------------------------------
# Synthetic demo sequence (no camera needed)
# ----------------------------------------------------------------------------------


def make_demo_video(
    out_path: Path,
    image_paths: list[Path],
    frames_per_image: int = 40,
    size: tuple[int, int] = (640, 480),
    fps: int = 25,
    backgrounds: str = "mixed",
    seed: int = 0,
) -> Path:
    """Animate still photos into a camera-like clip: the hand drifts, rotates and changes size,
    and (optionally) appears in front of cluttered backgrounds. Used to exercise and benchmark
    the streaming pipeline without a webcam."""
    from .landmarks import imread
    from .robustness import BackgroundBank, hand_mask

    rng = np.random.default_rng(seed)
    bank = BackgroundBank(seed=seed)
    W, H = size
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    with HandLandmarker(static_image_mode=True) as det:
        for idx, p in enumerate(image_paths):
            img = imread(p)
            d = det.detect(img)
            if d is None:
                continue
            alpha = hand_mask(img, d.landmarks)
            use_clutter = backgrounds == "clutter" or (backgrounds == "mixed" and idx % 2 == 1)
            if use_clutter:
                bg = cv2.resize(bank.sample(512), (W, H))
            else:
                border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
                bg = np.broadcast_to(np.median(border, axis=0), (H, W, 3)).astype(np.uint8)
            s0, s1 = rng.uniform(1.0, 1.25), rng.uniform(1.25, 1.6)
            a0, a1 = rng.uniform(-12, 12, 2)
            c0 = np.array([W / 2, H / 2]) + rng.uniform(-90, 90, 2)
            c1 = np.array([W / 2, H / 2]) + rng.uniform(-90, 90, 2)
            for f in range(frames_per_image):
                u = 0.5 - 0.5 * np.cos(np.pi * f / (frames_per_image - 1))  # ease in/out
                s, a, c = s0 + (s1 - s0) * u, a0 + (a1 - a0) * u, c0 + (c1 - c0) * u
                M = cv2.getRotationMatrix2D((128, 128), a, s)
                M[:, 2] += c - 128
                hand = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
                al = cv2.warpAffine(alpha, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                frame = (hand * al[..., None] + bg * (1 - al[..., None])).astype(np.uint8)
                noise = rng.normal(0, 3, frame.shape)
                writer.write(np.clip(frame + noise, 0, 255).astype(np.uint8))
    writer.release()
    return out_path
