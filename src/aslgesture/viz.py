"""Shared plotting helpers so every figure in ``reports/figures`` looks consistent.

Colour roles follow a validated, colour-blind-safe categorical order (blue,
orange, aqua) and a single-hue blue ramp for magnitudes (heatmaps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
NEUTRAL = "#b9b8b3"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def grouped_bars(
    ax,
    groups: Sequence[str],
    series: dict[str, Sequence[float]],
    value_fmt: str = "{:.1f}",
    colors: Sequence[str] | None = None,
    label_values: bool = True,
) -> None:
    """Grouped vertical bars with a thin surface gap between neighbours."""
    colors = list(colors or SERIES)
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (name, values) in enumerate(series.items()):
        pos = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(pos, values, width=width, color=colors[i % len(colors)], label=name,
                      edgecolor=SURFACE, linewidth=1.5)
        if label_values:
            for b, v in zip(bars, values):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), value_fmt.format(v),
                        ha="center", va="bottom", fontsize=7.5, color=TEXT_2)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)


def heatmap(ax, matrix: np.ndarray, row_labels, col_labels, fmt: str = "{:d}",
            annotate: bool = True, vmax: float | None = None) -> None:
    """Sequential single-hue heatmap; annotation ink flips to white on dark cells."""
    m = np.asarray(matrix, dtype=float)
    vmax = vmax if vmax is not None else (m.max() if m.size else 1.0)
    ax.imshow(m, cmap=SEQ_CMAP, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if annotate:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = matrix[i][j]
                if m[i, j] == 0:
                    continue
                ink = "white" if m[i, j] > 0.55 * vmax else TEXT
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5, color=ink)


def confusion(ax, cm: np.ndarray, labels: Sequence[str], title: str,
              col_labels: Sequence[str] | None = None) -> None:
    """Confusion matrix: colour = row-normalised rate, text = raw counts.

    ``col_labels`` allows extra outcome columns (e.g. "no hand detected")."""
    cm = np.asarray(cm)
    col_labels = list(col_labels or labels)
    rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(rates, cmap=SEQ_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(labels)), labels)
    if len(col_labels) > len(labels):
        ax.axvline(len(labels) - 0.5, color=TEXT_2, linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            ink = "white" if rates[i, j] > 0.55 else TEXT
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7.5, color=ink)
    ax.set_title(title)
