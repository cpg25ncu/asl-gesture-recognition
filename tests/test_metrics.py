"""End-to-end accuracy counts images without a detected hand as errors."""

import pytest

from aslgesture.supervised import end_to_end_accuracy


def test_undetected_images_count_as_errors():
    y_true = ["A", "B", "C", "D"]
    y_pred = ["A", "B", "C", "X"]  # 3 of 4 detected images correct
    assert end_to_end_accuracy(y_true, y_pred, n_undetected=0) == pytest.approx(0.75)
    assert end_to_end_accuracy(y_true, y_pred, n_undetected=2) == pytest.approx(0.5)  # 3 / 6
