"""The from-scratch kNN must agree with scikit-learn's reference implementation."""

import ast
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from aslgesture.knn import KNNClassifier, accuracy


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 8)) * 3
    X = np.vstack([c + rng.normal(size=(60, 8)) for c in centers])
    y = np.repeat(list("ABCD"), 60)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    return X[:180], y[:180], X[180:], y[180:]


@pytest.mark.parametrize("k", [1, 3, 7, 15])
@pytest.mark.parametrize("weighting", ["uniform", "distance"])
def test_matches_sklearn(data, k, weighting):
    X_tr, y_tr, X_te, _ = data
    ours = KNNClassifier(k=k, weighting=weighting).fit(X_tr, y_tr).predict(X_te)
    ref = KNeighborsClassifier(n_neighbors=k, weights=weighting, algorithm="brute").fit(X_tr, y_tr).predict(X_te)
    # Continuous random data has no distance ties, so predictions should be identical.
    assert ours == ref.tolist()


def test_predict_grid_equals_individual_models(data):
    X_tr, y_tr, X_te, _ = data
    grid = KNNClassifier(k=9).fit(X_tr, y_tr).predict_grid(X_te, ks=[1, 5, 9])
    for (k, w), preds in grid.items():
        assert preds == KNNClassifier(k=k, weighting=w).fit(X_tr, y_tr).predict(X_te)


def test_predict_proba_rows_sum_to_one(data):
    X_tr, y_tr, X_te, _ = data
    proba = KNNClassifier(k=5, weighting="distance").fit(X_tr, y_tr).predict_proba(X_te[:10])
    assert all(abs(sum(row) - 1.0) < 1e-9 for row in proba)


def test_uniform_tie_broken_by_distance():
    knn = KNNClassifier(k=2).fit([[0.0], [3.0]], ["far", "near"])
    assert knn.predict([[2.0]]) == ["near"]


def test_accuracy_and_validation():
    assert accuracy(["A", "B"], ["A", "C"]) == 0.5
    with pytest.raises(ValueError):
        KNNClassifier(k=0).fit([[0.0]], ["A"])
    with pytest.raises(ValueError):
        KNNClassifier(weighting="cosine").fit([[0.0]], ["A"])


def test_uses_only_standard_library():
    source = Path(__file__).resolve().parents[1] / "src" / "aslgesture" / "knn.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "heapq", "math", "collections", "typing"}
