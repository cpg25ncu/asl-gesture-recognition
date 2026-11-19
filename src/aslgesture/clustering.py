"""Unsupervised learning: K-Means and agglomerative (Ward) clustering on the landmark features.

Labels are never used to fit the clusterings. They are only used afterwards
to judge how well the discovered clusters line up with the ten ASL letters
(purity, adjusted Rand index, normalised mutual information) and to compare
the clusters with the predictions of the supervised classifiers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config, features, viz
from .supervised import load_dataset, landmark_array

K_RANGE = range(2, 21)
METHODS = {
    "K-Means": lambda k: KMeans(n_clusters=k, n_init=10, random_state=config.SEED),
    "Agglomerative (Ward)": lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward"),
}


def purity(y_true: np.ndarray, clusters: np.ndarray) -> float:
    ct = pd.crosstab(clusters, y_true)
    return float(ct.max(axis=1).sum() / ct.to_numpy().sum())


def majority_mapping(y_true: np.ndarray, clusters: np.ndarray) -> dict[int, str]:
    """Map each cluster to the letter that occurs most often inside it."""
    ct = pd.crosstab(clusters, y_true)
    return ct.idxmax(axis=1).to_dict()


def run(feature_sets=("raw", "norm")) -> pd.DataFrame:
    config.ensure_dirs()
    viz.setup()
    df, _ = load_dataset()
    y = df["label"].to_numpy()
    lm = landmark_array(df)

    curves, rows, assignments = [], [], {}
    for fs in feature_sets:
        X = StandardScaler().fit_transform(features.build_features(lm, fs))
        for name, make in METHODS.items():
            for k in K_RANGE:
                model = make(k)
                labels = model.fit_predict(X)
                curves.append({"feature_set": fs, "method": name, "k": k,
                               "silhouette": silhouette_score(X, labels),
                               "inertia": getattr(model, "inertia_", np.nan)})
                if k == len(config.LABELS):
                    assignments[(fs, name)] = labels
                    rows.append({"feature_set": fs, "method": name, "k": k,
                                 "silhouette": round(silhouette_score(X, labels), 4),
                                 "purity": round(purity(y, labels), 4),
                                 "ARI": round(adjusted_rand_score(y, labels), 4),
                                 "NMI": round(normalized_mutual_info_score(y, labels), 4)})
            print(f"[{fs}] {name} done")

    curves_df = pd.DataFrame(curves)
    curves_df.to_csv(config.TABLE_DIR / "clustering_curves.csv", index=False)
    summary = pd.DataFrame(rows)

    # Compare clusters with the supervised models on the test set.
    pred_path = config.PROCESSED_DIR / "test_predictions.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path)
        test_mask = df["instance_id"].isin(preds["instance_id"]).to_numpy()
        test_ids = df.loc[test_mask, "instance_id"]
        preds = preds.set_index("instance_id").loc[test_ids]
        best_col = _best_prediction_column()
        for i, r in summary.iterrows():
            labels = assignments[(r["feature_set"], r["method"])]
            mapping = majority_mapping(y, labels)
            mapped = np.array([mapping[c] for c in labels[test_mask]])
            summary.loc[i, "test_acc_majority_label"] = round(float((mapped == y[test_mask]).mean()), 4)
            if best_col:
                summary.loc[i, "agreement_with_best_classifier"] = round(
                    float((mapped == preds[best_col].to_numpy()).mean()), 4)
    summary.to_csv(config.TABLE_DIR / "clustering_summary.csv", index=False)

    _plot_silhouette(curves_df, feature_sets)
    best = summary.sort_values("ARI", ascending=False).iloc[0]
    _plot_crosstab(y, assignments[(best["feature_set"], best["method"])],
                   f"{best['method']} (k=10, {best['feature_set']} features) vs true letters")
    _plot_pca_small_multiples(StandardScaler().fit_transform(features.build_features(lm, "norm")), y)
    return summary


def _best_prediction_column() -> str | None:
    path = config.TABLE_DIR / "supervised_summary.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path).sort_values("cv_acc_mean", ascending=False).iloc[0]
    key = {"kNN (from scratch)": "knn", "Decision Tree": "dt", "Logistic Regression": "lr"}[s["model"]]
    return f"pred_{key}_{s['feature_set']}"


def _plot_silhouette(curves: pd.DataFrame, feature_sets) -> None:
    fig, axes = viz.plt.subplots(1, len(feature_sets), figsize=(5.2 * len(feature_sets), 3.8), sharey=True)
    for ax, fs in zip(np.atleast_1d(axes), feature_sets):
        for i, name in enumerate(METHODS):
            d = curves[(curves["feature_set"] == fs) & (curves["method"] == name)]
            ax.plot(d["k"], d["silhouette"], marker="o", color=viz.SERIES[i], label=name)
        ax.axvline(len(config.LABELS), color=viz.NEUTRAL, linestyle="--", linewidth=1)
        ax.text(len(config.LABELS) + 0.2, ax.get_ylim()[1] * 0.97, "k = 10 letters", fontsize=8,
                color=viz.TEXT_2, va="top")
        ax.set_xlabel("number of clusters k")
        ax.set_title(f"Silhouette score - {fs} features")
    np.atleast_1d(axes)[0].set_ylabel("silhouette")
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout()
    viz.save(fig, config.FIG_DIR / "clustering_silhouette.png")


def _plot_crosstab(y: np.ndarray, clusters: np.ndarray, title: str) -> None:
    ct = pd.crosstab(clusters, y).reindex(columns=config.LABELS, fill_value=0)
    ct = ct.loc[ct.idxmax(axis=1).map(config.LABELS.index).sort_values(kind="stable").index]
    fig, ax = viz.plt.subplots(figsize=(6.4, 4.8))
    viz.heatmap(ax, ct.to_numpy(), [f"cluster {c}" for c in ct.index], config.LABELS)
    ax.set_xlabel("true letter")
    ax.set_title(title, fontsize=10.5)
    fig.tight_layout()
    viz.save(fig, config.FIG_DIR / "clustering_crosstab.png")


def _plot_pca_small_multiples(X: np.ndarray, y: np.ndarray) -> None:
    """One panel per letter: that letter in blue, everything else in light grey."""
    pc = PCA(n_components=2, random_state=0).fit_transform(X)
    fig, axes = viz.plt.subplots(2, 5, figsize=(13, 5.4), sharex=True, sharey=True)
    for ax, letter in zip(axes.ravel(), config.LABELS):
        m = y == letter
        ax.scatter(pc[~m, 0], pc[~m, 1], s=3, color="#dcdbd6", linewidths=0)
        ax.scatter(pc[m, 0], pc[m, 1], s=4, color=viz.SERIES[0], linewidths=0)
        ax.set_title(letter, loc="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("PCA projection of normalised landmarks, one panel per letter", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=viz.TEXT)
    fig.tight_layout()
    viz.save(fig, config.FIG_DIR / "pca_by_letter.png")
