"""k-Nearest Neighbours implemented from scratch with the Python standard library only.

No NumPy / scikit-learn is used here on purpose: the classifier is written
with ``math``, ``heapq`` and ``collections`` so that every step (distance,
neighbour search, voting, tie-breaking) is explicit.

Speed-ups that keep it pure Python:

* ``math.dist`` computes Euclidean distance in C.
* ``heapq.nsmallest`` finds the k nearest neighbours in O(n log k) instead of
  sorting all n distances.
* ``predict_grid`` computes neighbours once for the largest k and re-uses them
  for every (k, weighting) combination, which makes 5-fold cross-validation
  over the whole grid as cheap as a single model evaluation.

The class follows the scikit-learn estimator interface (``fit``, ``predict``,
``predict_proba``, ``score``, ``get_params``/``set_params``) so it plugs into
the same evaluation code as the library models.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Iterable, Sequence

WEIGHTINGS = ("uniform", "distance")


def _as_rows(X) -> list[list[float]]:
    """Accept lists or array-likes (anything with ``tolist``) without importing NumPy."""
    if hasattr(X, "tolist"):
        X = X.tolist()
    return [list(map(float, row)) for row in X]


def _as_labels(y) -> list[str]:
    if hasattr(y, "tolist"):
        y = y.tolist()
    return [str(v) for v in y]


def _vote(neighbours: Sequence[tuple[float, str]], weighting: str, eps: float) -> dict[str, float]:
    """Return the (unnormalised) vote mass per label for a list of (distance, label)."""
    votes: dict[str, float] = defaultdict(float)
    for dist, label in neighbours:
        votes[label] += 1.0 if weighting == "uniform" else 1.0 / (dist + eps)
    return votes


def _decide(votes: dict[str, float], neighbours: Sequence[tuple[float, str]]) -> str:
    """Highest vote wins; ties are broken by the smaller summed distance, then alphabetically."""
    best = max(votes.values())
    tied = [lbl for lbl, v in votes.items() if v == best]
    if len(tied) == 1:
        return tied[0]
    dist_sum: dict[str, float] = defaultdict(float)
    for dist, lbl in neighbours:
        if lbl in tied:
            dist_sum[lbl] += dist
    return min(tied, key=lambda lbl: (dist_sum[lbl], lbl))


class KNNClassifier:
    """kNN classifier with uniform or inverse-distance weighted voting."""

    def __init__(self, k: int = 5, weighting: str = "uniform", eps: float = 1e-9) -> None:
        self.k = k
        self.weighting = weighting
        self.eps = eps

    # -- scikit-learn compatibility -------------------------------------------------
    def get_params(self, deep: bool = True) -> dict:
        return {"k": self.k, "weighting": self.weighting, "eps": self.eps}

    def set_params(self, **params) -> "KNNClassifier":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return f"KNNClassifier(k={self.k}, weighting={self.weighting!r})"

    # -- core -----------------------------------------------------------------------
    def fit(self, X, y) -> "KNNClassifier":
        X_rows, labels = _as_rows(X), _as_labels(y)
        if not X_rows:
            raise ValueError("X is empty.")
        if len(X_rows) != len(labels):
            raise ValueError("X and y must have the same length.")
        if self.k <= 0:
            raise ValueError("k must be positive.")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}.")
        self._X = X_rows
        self._y = labels
        self.classes_ = sorted(set(labels))
        return self

    def kneighbors(self, x: Sequence[float], k: int | None = None) -> list[tuple[float, str]]:
        """Return the k nearest ``(distance, label)`` pairs, closest first."""
        k = self.k if k is None else k
        dist = math.dist
        return heapq.nsmallest(k, ((dist(x, xi), yi) for xi, yi in zip(self._X, self._y)), key=lambda t: t[0])

    def _predict_one(self, x: Sequence[float]) -> str:
        neighbours = self.kneighbors(x)
        return _decide(_vote(neighbours, self.weighting, self.eps), neighbours)

    def predict(self, X) -> list[str]:
        return [self._predict_one(x) for x in _as_rows(X)]

    def predict_proba(self, X) -> list[list[float]]:
        """Vote share per class (columns ordered as ``classes_``)."""
        out = []
        for x in _as_rows(X):
            votes = _vote(self.kneighbors(x), self.weighting, self.eps)
            total = sum(votes.values())
            out.append([votes.get(c, 0.0) / total for c in self.classes_])
        return out

    def score(self, X, y) -> float:
        return accuracy(_as_labels(y), self.predict(X))

    def predict_grid(
        self, X, ks: Iterable[int], weightings: Iterable[str] = WEIGHTINGS
    ) -> dict[tuple[int, str], list[str]]:
        """Predict for every (k, weighting) pair while searching neighbours only once."""
        ks, weightings = sorted(set(ks)), list(weightings)
        preds: dict[tuple[int, str], list[str]] = {(k, w): [] for k in ks for w in weightings}
        for x in _as_rows(X):
            neighbours = self.kneighbors(x, k=ks[-1])
            for k in ks:
                nk = neighbours[:k]
                for w in weightings:
                    preds[(k, w)].append(_decide(_vote(nk, w, self.eps), nk))
        return preds


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
