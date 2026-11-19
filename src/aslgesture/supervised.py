"""Supervised learning: cleaning, train/test split, 5-fold CV tuning and final evaluation.

Protocol
--------
1. A stratified 70/30 train/test split is created once and saved to
   ``split.csv`` so every model and feature set sees exactly the same images.
2. For each model, every hyper-parameter combination in its grid (which always
   includes the library default as a baseline) is scored with stratified
   5-fold cross-validation on the *training set only*.
3. The best combination (highest mean CV accuracy, ties -> lower std) is refit
   on the full training set and evaluated once on the held-out test set.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config, features, viz
from .knn import KNNClassifier
from .landmarks import COORD_COLUMNS

# ----------------------------------------------------------------------------------
# Models and hyper-parameter grids
# ----------------------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    factory: Callable[..., object]
    grid: dict
    default: dict
    parametric: bool = True  # False for instance-based models that store the training set


def _logreg(C: float = 1.0, l1_ratio: float = 0.0):
    # Standardising first keeps the saga solver well conditioned for every C.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="saga", C=C, l1_ratio=l1_ratio, max_iter=5000, random_state=config.SEED),
    )


MODELS: dict[str, ModelSpec] = {
    "knn": ModelSpec(
        name="kNN (from scratch)",
        factory=lambda **p: KNNClassifier(**p),
        grid={"k": [1, 3, 5, 7, 9, 11, 15], "weighting": ["uniform", "distance"]},
        default={"k": 5, "weighting": "uniform"},
        parametric=False,
    ),
    "dt": ModelSpec(
        name="Decision Tree",
        factory=lambda **p: DecisionTreeClassifier(random_state=config.SEED, **p),
        grid={"max_depth": [None, 5, 10, 15, 20, 30], "min_samples_split": [2, 5, 10, 20]},
        default={"max_depth": None, "min_samples_split": 2},
    ),
    "lr": ModelSpec(
        name="Logistic Regression",
        factory=_logreg,
        grid={"C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], "l1_ratio": [0.0, 1.0]},  # 0 = L2, 1 = L1
        default={"C": 1.0, "l1_ratio": 0.0},
    ),
}

# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------


def load_landmarks() -> pd.DataFrame:
    if not config.LANDMARKS_CSV.exists():
        raise FileNotFoundError(f"{config.LANDMARKS_CSV} not found - run `asl-gesture extract` first.")
    return pd.read_csv(config.LANDMARKS_CSV)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop incomplete rows, duplicated IDs and exact duplicate landmark vectors."""
    report = {"rows_in": len(df)}
    df = df.dropna(subset=COORD_COLUMNS + ["label"])
    report["after_dropna"] = len(df)
    df = df.drop_duplicates(subset="instance_id")
    report["after_dedup_id"] = len(df)
    df = df.drop_duplicates(subset=COORD_COLUMNS)
    report["after_dedup_landmarks"] = len(df)
    return df.reset_index(drop=True), report


def get_split(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with a ``split`` column; the split is created once and then re-used."""
    if config.SPLIT_CSV.exists():
        split = pd.read_csv(config.SPLIT_CSV)
        if set(split["instance_id"]) == set(df["instance_id"]):
            return df.merge(split, on="instance_id")
    train_ids, test_ids = train_test_split(
        df["instance_id"], test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.SEED
    )
    split = pd.concat(
        [pd.DataFrame({"instance_id": train_ids, "split": "train"}),
         pd.DataFrame({"instance_id": test_ids, "split": "test"})]
    )
    split.to_csv(config.SPLIT_CSV, index=False)
    return df.merge(split, on="instance_id")


def undetected_split() -> pd.DataFrame:
    """Train/test assignment for the images on which MediaPipe found no hand.

    These images never reach a classifier, so assigning them after the fact does
    not change any trained model. They are split with the same ratio, stratified
    by letter and with the same seed, so that the test set represents *every*
    image and end-to-end (full-system) accuracy can be reported. Byte-identical
    duplicates are removed, consistent with the cleaning of detected images.
    """
    path = config.PROCESSED_DIR / "split_undetected.csv"
    if path.exists():
        return pd.read_csv(path)
    import hashlib

    ids, seen = [], set()
    for line in config.FAILURES_TXT.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel:
            continue
        digest = hashlib.md5((config.RAW_DIR / rel).read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            ids.append(rel)
    labels = [rel.split("/")[0] for rel in ids]
    _, test_ids = train_test_split(ids, test_size=config.TEST_SIZE, stratify=labels, random_state=config.SEED)
    test_ids = set(test_ids)
    out = pd.DataFrame({"instance_id": ids, "label": labels,
                        "split": ["test" if i in test_ids else "train" for i in ids]})
    out.to_csv(path, index=False)
    return out


def end_to_end_accuracy(y_true, y_pred, n_undetected: int) -> float:
    """Accuracy over all test images: images without a detected hand count as errors."""
    correct = int(np.sum(np.asarray(y_true) == np.asarray(y_pred)))
    return correct / (len(y_true) + n_undetected)


def landmark_array(df: pd.DataFrame) -> np.ndarray:
    return df[COORD_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, config.N_LANDMARKS, 3)


def load_dataset() -> tuple[pd.DataFrame, dict]:
    """Cleaned landmarks with a fixed ``split`` column."""
    df, report = clean(load_landmarks())
    return get_split(df), report


# ----------------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------------


def _folds(y: np.ndarray):
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    return list(skf.split(np.zeros(len(y)), y))


def _param_key(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _train_part(X, y, idx, augment: bool, seed: int = 0):
    """Training rows of one fold; augmentation is applied *after* the split so that
    rotated copies of a validation sample never appear in its training fold."""
    if not augment:
        return X[idx], y[idx]
    return features.augment_rotations(X[idx], y[idx], seed=seed)


def cross_validate(key: str, X: np.ndarray, y: np.ndarray, augment: bool = False) -> pd.DataFrame:
    """Score every grid point with stratified 5-fold CV. Returns one row per setting."""
    spec = MODELS[key]
    folds = _folds(y)
    scores: dict[str, dict] = {}

    if key == "knn":
        # Neighbours are searched once per fold and re-used for every (k, weighting).
        ks, ws = spec.grid["k"], spec.grid["weighting"]
        for i, (tr, va) in enumerate(folds):
            knn = KNNClassifier(k=max(ks)).fit(*_train_part(X, y, tr, augment, seed=i))
            preds = knn.predict_grid(X[va], ks, ws)
            for (k, w), p in preds.items():
                params = {"k": k, "weighting": w}
                s = scores.setdefault(_param_key(params), {"params": params, "acc": [], "f1": []})
                s["acc"].append(accuracy_score(y[va], p))
                s["f1"].append(f1_score(y[va], p, average="macro"))
    else:
        from joblib import Parallel, delayed

        def run(params):
            accs, f1s = [], []
            for i, (tr, va) in enumerate(folds):
                model = spec.factory(**params).fit(*_train_part(X, y, tr, augment, seed=i))
                p = model.predict(X[va])
                accs.append(accuracy_score(y[va], p))
                f1s.append(f1_score(y[va], p, average="macro"))
            return params, accs, f1s

        for params, accs, f1s in Parallel(n_jobs=-1)(delayed(run)(p) for p in ParameterGrid(spec.grid)):
            scores[_param_key(params)] = {"params": params, "acc": accs, "f1": f1s}

    rows = []
    for s in scores.values():
        row = dict(s["params"])
        row["is_default"] = s["params"] == spec.default
        row["cv_acc_mean"] = float(np.mean(s["acc"]))
        row["cv_acc_std"] = float(np.std(s["acc"]))
        row["cv_f1_mean"] = float(np.mean(s["f1"]))
        for i, a in enumerate(s["acc"], 1):
            row[f"fold{i}_acc"] = float(a)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["cv_acc_mean", "cv_acc_std"], ascending=[False, True])
    return out.reset_index(drop=True)


def best_params(cv: pd.DataFrame, key: str) -> dict:
    row = cv.iloc[0]
    params = {}
    for k in MODELS[key].grid:
        v = row[k]
        if isinstance(v, float) and np.isnan(v):
            v = None
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
            if k in ("max_depth", "min_samples_split", "k"):
                v = int(v)
        params[k] = v
    return params


# ----------------------------------------------------------------------------------
# Final evaluation
# ----------------------------------------------------------------------------------


@dataclass
class FinalResult:
    key: str
    feature_set: str
    params: dict
    model: object
    cv_acc_mean: float
    cv_acc_std: float
    train_acc: float
    test_acc: float
    test_f1: float
    default_test_acc: float
    predict_ms: float
    y_pred: np.ndarray = field(repr=False, default=None)


def _timed_predict_ms(model, X: np.ndarray, n: int = 200) -> float:
    """Median single-sample prediction latency in milliseconds (as used per video frame)."""
    idx = np.random.default_rng(0).choice(len(X), size=min(n, len(X)), replace=False)
    times = []
    for i in idx:
        t0 = time.perf_counter()
        model.predict(X[i : i + 1])
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


def evaluate(key: str, feature_set: str, cv: pd.DataFrame, X_tr, y_tr, X_te, y_te) -> FinalResult:
    spec = MODELS[key]
    params = best_params(cv, key)
    fit_data = (features.augment_rotations(X_tr, y_tr, seed=99) if feature_set in features.AUGMENTED_SETS
                else (X_tr, y_tr))
    model = spec.factory(**params).fit(*fit_data)
    y_pred = np.asarray(model.predict(X_te))
    default_model = spec.factory(**spec.default).fit(*fit_data)
    return FinalResult(
        key=key,
        feature_set=feature_set,
        params=params,
        model=model,
        cv_acc_mean=float(cv.iloc[0]["cv_acc_mean"]),
        cv_acc_std=float(cv.iloc[0]["cv_acc_std"]),
        train_acc=float(accuracy_score(y_tr, model.predict(X_tr))),
        test_acc=float(accuracy_score(y_te, y_pred)),
        test_f1=float(f1_score(y_te, y_pred, average="macro")),
        default_test_acc=float(accuracy_score(y_te, default_model.predict(X_te))),
        predict_ms=_timed_predict_ms(model, X_te),
        y_pred=y_pred,
    )


# ----------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------


def plot_cv(cvs: dict[str, pd.DataFrame], feature_set: str) -> None:
    viz.setup()
    fig, axes = viz.plt.subplots(1, 3, figsize=(15, 4.2))

    # kNN: accuracy vs k, one line per voting scheme
    ax = axes[0]
    knn = cvs["knn"]
    for i, w in enumerate(MODELS["knn"].grid["weighting"]):
        d = knn[knn["weighting"] == w].sort_values("k")
        ax.errorbar(d["k"], d["cv_acc_mean"] * 100, yerr=d["cv_acc_std"] * 100, marker="o",
                    color=viz.SERIES[i], label=f"{w} vote", capsize=3)
    ax.set_xlabel("k (neighbours)")
    ax.set_ylabel("5-fold CV accuracy (%)")
    ax.set_title("kNN (from scratch)")
    ax.legend()

    # Decision tree: heatmap max_depth x min_samples_split
    ax = axes[1]
    dt = cvs["dt"].copy()
    dt["max_depth"] = dt["max_depth"].apply(lambda v: "None" if pd.isna(v) else str(int(v)))
    depths = [("None" if d is None else str(d)) for d in MODELS["dt"].grid["max_depth"]]
    splits = MODELS["dt"].grid["min_samples_split"]
    mat = np.array([[dt[(dt["max_depth"] == d) & (dt["min_samples_split"] == s)]["cv_acc_mean"].iloc[0] * 100
                     for s in splits] for d in depths])
    # Colour scale covers the top 8 points so small differences stay visible;
    # anything lower (e.g. a badly under-fitting depth) is clipped to the lightest step.
    lo = max(np.floor(mat.min()), np.ceil(mat.max()) - 8)
    shade = np.clip(mat - lo, 0, None)
    viz.heatmap(ax, shade, depths, [str(s) for s in splits], annotate=False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ink = "white" if shade[i, j] > 0.55 * shade.max() else viz.TEXT
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8, color=ink)
    ax.set_xlabel("min_samples_split")
    ax.set_ylabel("max_depth")
    ax.set_title("Decision Tree (CV accuracy %)")

    # Logistic regression: accuracy vs C, one line per penalty
    ax = axes[2]
    lr = cvs["lr"]
    for i, (r, name) in enumerate([(0.0, "L2"), (1.0, "L1")]):
        d = lr[lr["l1_ratio"] == r].sort_values("C")
        ax.errorbar(d["C"], d["cv_acc_mean"] * 100, yerr=d["cv_acc_std"] * 100, marker="o",
                    color=viz.SERIES[i], label=f"{name} penalty", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse regularisation strength)")
    ax.set_ylabel("5-fold CV accuracy (%)")
    ax.set_title("Logistic Regression")
    ax.legend()

    fig.suptitle(f"Hyper-parameter tuning - feature set: {feature_set}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=viz.TEXT)
    fig.tight_layout()
    viz.save(fig, config.FIG_DIR / f"cv_tuning_{feature_set}.png")


def plot_feature_set_comparison(summary: pd.DataFrame) -> None:
    viz.setup()
    fig, ax = viz.plt.subplots(figsize=(8, 4.2))
    models = [MODELS[k].name for k in MODELS]
    series = {}
    for fs in summary["feature_set"].unique():
        d = summary[summary["feature_set"] == fs].set_index("model")
        series[fs] = [d.loc[m, "test_acc"] * 100 for m in models]
    viz.grouped_bars(ax, models, series, value_fmt="{:.1f}")
    lo = min(min(v) for v in series.values())
    ax.set_ylim(max(0, np.floor(lo / 5) * 5 - 5), 100)
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_title("Test accuracy on images with a detected hand, by model and feature set", fontsize=11)
    ax.legend(title="feature set", loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    viz.save(fig, config.FIG_DIR / "feature_set_comparison.png")


def plot_confusion(key: str, feature_set: str, y_te, y_pred, undetected_labels=()) -> None:
    """Full-system confusion matrix: test images MediaPipe missed appear in a "no hand" column."""
    viz.setup()
    cm = confusion_matrix(y_te, y_pred, labels=config.LABELS)
    missed = pd.Series(list(undetected_labels), dtype=object).value_counts().reindex(config.LABELS, fill_value=0)
    cm = np.column_stack([cm, missed.to_numpy()])
    cls_acc = float(np.mean(np.asarray(y_te) == np.asarray(y_pred)))
    e2e = end_to_end_accuracy(y_te, y_pred, len(undetected_labels))
    fig, ax = viz.plt.subplots(figsize=(5.8, 4.9))
    viz.confusion(ax, cm, config.LABELS, f"{MODELS[key].name} | {feature_set}\n"
                  f"classification {cls_acc:.1%}  ·  end-to-end {e2e:.1%}", col_labels=[*config.LABELS, "no\nhand"])
    fig.tight_layout()
    viz.save(fig, config.FIG_DIR / f"confusion_{key}_{feature_set}.png")


def plot_class_distribution(df_all: pd.DataFrame, n_images_per_class: dict[str, int]) -> None:
    viz.setup()
    fig, ax = viz.plt.subplots(figsize=(8, 3.6))
    detected = df_all["label"].value_counts().reindex(config.LABELS, fill_value=0)
    viz.grouped_bars(ax, config.LABELS,
                     {"images": [n_images_per_class.get(c, 0) for c in config.LABELS],
                      "hand detected (kept)": detected.tolist()}, value_fmt="{:.0f}")
    ax.set_ylabel("Count")
    ax.set_title("Class distribution before and after MediaPipe detection / cleaning")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    viz.save(fig, config.FIG_DIR / "class_distribution.png")


# ----------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------


def run(feature_sets=features.FEATURE_SETS, model_keys=tuple(MODELS)) -> pd.DataFrame:
    config.ensure_dirs()
    df, clean_report = load_dataset()
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()
    und = undetected_split()
    und_test = und[und["split"] == "test"]["label"].tolist()
    print(f"Cleaning: {clean_report}")
    print(f"Split: train={len(train)}  test={len(test)}  (+{len(und_test)} test images with no hand detected)")
    (config.TABLE_DIR / "cleaning_report.json").write_text(json.dumps({
        **clean_report, "train": len(train), "test": len(test),
        "undetected_unique": len(und), "undetected_train": int((und["split"] == "train").sum()),
        "undetected_test": len(und_test), "test_all_images": len(test) + len(und_test),
    }, indent=2))

    from .landmarks import list_dataset_images
    counts: dict[str, int] = {}
    for label, _ in list_dataset_images(config.RAW_DIR):
        counts[label] = counts.get(label, 0) + 1
    if counts:
        plot_class_distribution(df, counts)

    rows, results = [], {}
    for fs in feature_sets:
        X_tr = features.build_features(landmark_array(train), fs)
        X_te = features.build_features(landmark_array(test), fs)
        cvs = {}
        for key in model_keys:
            t0 = time.perf_counter()
            cv = cross_validate(key, X_tr, y_tr, augment=fs in features.AUGMENTED_SETS)
            cv.to_csv(config.TABLE_DIR / f"cv_{key}_{fs}.csv", index=False)
            cvs[key] = cv
            res = evaluate(key, fs, cv, X_tr, y_tr, X_te, y_te)
            results[(key, fs)] = res
            print(f"[{fs:8s}] {MODELS[key].name:22s} best={_param_key(res.params):35s} "
                  f"cv={res.cv_acc_mean:.4f}±{res.cv_acc_std:.4f} test={res.test_acc:.4f} "
                  f"({time.perf_counter() - t0:.0f}s)")
            rows.append({
                "feature_set": fs, "model": MODELS[key].name, "best_params": _param_key(res.params),
                "cv_acc_mean": round(res.cv_acc_mean, 5), "cv_acc_std": round(res.cv_acc_std, 5),
                "train_acc": round(res.train_acc, 5), "test_acc": round(res.test_acc, 5),
                "test_acc_end_to_end": round(end_to_end_accuracy(y_te, res.y_pred, len(und_test)), 5),
                "test_f1_macro": round(res.test_f1, 5), "default_params_test_acc": round(res.default_test_acc, 5),
                "predict_ms_per_sample": round(res.predict_ms, 3),
            })
            plot_confusion(key, fs, y_te, res.y_pred, und_test)
            report = classification_report(y_te, res.y_pred, labels=config.LABELS, output_dict=True, zero_division=0)
            pd.DataFrame(report).T.to_csv(config.TABLE_DIR / f"per_class_{key}_{fs}.csv")
        if set(cvs) == set(MODELS):
            plot_cv(cvs, fs)

    # Merge with results of earlier runs for other feature sets / models.
    summary_path = config.TABLE_DIR / "supervised_summary.csv"
    summary = pd.DataFrame(rows)
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        keep = ~old.set_index(["feature_set", "model"]).index.isin(summary.set_index(["feature_set", "model"]).index)
        summary = pd.concat([old[keep], summary], ignore_index=True)
    order = {fs: i for i, fs in enumerate(features.FEATURE_SETS)}
    summary = summary.sort_values("feature_set", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
    summary.to_csv(summary_path, index=False)
    if len(summary["feature_set"].unique()) > 1:
        plot_feature_set_comparison(summary)

    # Persist every final model (local only: kNN models contain training data).
    store = config.PROCESSED_DIR / "models"
    store.mkdir(parents=True, exist_ok=True)
    for (key, fs), res in results.items():
        joblib.dump({"model": res.model, "feature_set": fs, "key": key, "params": res.params},
                    store / f"{key}_{fs}.joblib")
    pred_path = config.PROCESSED_DIR / "test_predictions.csv"
    preds = pd.DataFrame({"instance_id": test["instance_id"].to_numpy(), "label": y_te,
                          **{f"pred_{k}_{fs}": r.y_pred for (k, fs), r in results.items()}})
    if pred_path.exists():
        old = pd.read_csv(pred_path)
        extra = [c for c in old.columns if c.startswith("pred_") and c not in preds.columns]
        preds = preds.merge(old[["instance_id", *extra]], on="instance_id", how="left")
    preds.to_csv(pred_path, index=False)
    return summary


def export_model(key: str, feature_set: str, path=config.DEFAULT_MODEL) -> dict:
    """Copy one locally trained final model into ``models/`` for the demo / predict commands."""
    src = config.PROCESSED_DIR / "models" / f"{key}_{feature_set}.joblib"
    bundle = joblib.load(src)
    meta = {"model_name": MODELS[key].name, "key": key, "feature_set": feature_set,
            "params": bundle["params"], "labels": config.LABELS}
    joblib.dump({"model": bundle["model"], **meta}, path)
    (path.with_suffix(".json")).write_text(json.dumps(meta, indent=2, default=str))
    return meta
