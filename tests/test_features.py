"""Normalised features must be invariant to where the hand is, how big it is and (optionally) its rotation."""

import numpy as np
import pytest

from aslgesture.features import WRIST, MIDDLE_MCP, build_features, normalize_landmarks, to_isotropic


@pytest.fixture
def hand():
    rng = np.random.default_rng(1)
    lm = rng.uniform(0.3, 0.7, size=(21, 3))
    lm[:, 2] *= 0.1
    return lm


def rotate_xy(lm, angle):
    c, s = np.cos(angle), np.sin(angle)
    out = lm.copy()
    out[:, 0] = c * lm[:, 0] - s * lm[:, 1]
    out[:, 1] = s * lm[:, 0] + c * lm[:, 1]
    return out


def test_translation_and_scale_invariance(hand):
    moved = hand * 0.5 + np.array([0.2, -0.1, 0.0])
    np.testing.assert_allclose(build_features(hand, "norm"), build_features(moved, "norm"), atol=1e-9)


def test_raw_features_are_not_invariant(hand):
    moved = hand + np.array([0.2, 0.0, 0.0])
    assert not np.allclose(build_features(hand, "raw"), build_features(moved, "raw"))


def test_rotation_invariance_only_with_rotation_alignment(hand):
    rotated = rotate_xy(hand, np.deg2rad(35))
    np.testing.assert_allclose(build_features(hand, "norm_rot"), build_features(rotated, "norm_rot"), atol=1e-9)
    assert not np.allclose(build_features(hand, "norm"), build_features(rotated, "norm"))


def test_normalised_geometry(hand):
    pts = normalize_landmarks(hand, rotate=True)
    np.testing.assert_allclose(pts[WRIST], 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(pts[MIDDLE_MCP]), 1.0)
    assert pts[MIDDLE_MCP, 1] < 0 and abs(pts[MIDDLE_MCP, 0]) < 1e-9  # palm axis points up


def test_batch_shape(hand):
    batch = np.stack([hand, hand * 0.9])
    for fs in ("raw", "norm", "norm_rot"):
        assert build_features(batch, fs).shape == (2, 63)


def test_isotropic_fixes_aspect_ratio():
    # A square drawn on a 640x480 frame is not square in MediaPipe's normalised coordinates.
    sq = np.zeros((21, 3))
    sq[1:5, :2] = [[0.6, 0.5], [0.6, 0.6], [0.5, 0.6], [0.5, 0.5]]
    sq[1:5, 0] = 0.5 + (sq[1:5, 0] - 0.5) * 480 / 640  # 0.1 of height == 0.075 of width
    iso = to_isotropic(sq, 640, 480)[0]
    width = iso[1:5, 0].max() - iso[1:5, 0].min()
    height = iso[1:5, 1].max() - iso[1:5, 1].min()
    assert width == pytest.approx(height)
