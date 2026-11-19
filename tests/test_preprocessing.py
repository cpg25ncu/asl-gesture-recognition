"""Image pre-processing and training-time augmentation."""

import numpy as np

from aslgesture.features import augment_rotations, build_features, rotate_xy
from aslgesture.robustness import auto_gamma, denoise


def test_auto_gamma_brightens_dark_images_only():
    dark = np.full((32, 32, 3), 40, np.uint8)
    bright = np.full((32, 32, 3), 220, np.uint8)
    assert auto_gamma(dark).mean() > 100
    assert np.array_equal(auto_gamma(bright), bright)  # never darkens


def test_denoise_reduces_noise():
    rng = np.random.default_rng(0)
    clean = np.full((64, 64, 3), 120, np.uint8)
    noisy = np.clip(clean + rng.normal(0, 15, clean.shape), 0, 255).astype(np.uint8)
    assert denoise(noisy).astype(float).std() < noisy.astype(float).std() / 2


def test_rotation_augmentation_preserves_hand_size():
    rng = np.random.default_rng(2)
    X = build_features(rng.uniform(0.3, 0.7, size=(5, 21, 3)), "norm")
    Xa, ya = augment_rotations(X, np.arange(5), copies=3)
    assert Xa.shape == (20, 63) and list(ya[:10]) == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    np.testing.assert_allclose(np.linalg.norm(Xa.reshape(-1, 21, 3), axis=2),
                               np.tile(np.linalg.norm(X.reshape(-1, 21, 3), axis=2), (4, 1)), atol=1e-9)
    np.testing.assert_allclose(rotate_xy(rotate_xy(X, 25.0), -25.0), X, atol=1e-9)
