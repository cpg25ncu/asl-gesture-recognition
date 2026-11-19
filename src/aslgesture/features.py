"""Feature engineering on the 21 x 3 hand landmarks returned by MediaPipe.

MediaPipe returns x and y normalised by the image width and height, so the
raw 63-value vector encodes *where* the hand is in the frame and *how big* it
appears as much as *what shape* it makes. That is fine for tightly cropped,
centred images, but breaks as soon as the hand moves around a webcam frame.

Feature sets
------------
``raw``        the 63 coordinates exactly as MediaPipe returns them.
``norm``       translation + scale invariant: wrist moved to the origin and
               coordinates divided by the palm length (wrist -> middle-finger MCP).
``norm_rot``   ``norm`` plus in-plane rotation alignment (palm axis rotated to
               point straight up).
``norm_aug``   same features as ``norm`` at inference time; during training the
               set is enlarged with randomly rotated copies (see
               ``augment_rotations``) so the model learns rotation tolerance
               without discarding orientation, which distinguishes I from J.
"""

from __future__ import annotations

import numpy as np

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17
FEATURE_SETS = ("raw", "norm", "norm_rot", "norm_aug")
AUGMENTED_SETS = {"norm_aug"}


def rotate_xy(features63: np.ndarray, degrees: np.ndarray | float) -> np.ndarray:
    """Rotate normalised (wrist-centred) feature vectors in the image plane."""
    pts = np.asarray(features63, dtype=np.float64).reshape(-1, 21, 3).copy()
    t = np.deg2rad(np.broadcast_to(np.asarray(degrees, dtype=np.float64).reshape(-1), (len(pts),)))
    c, s = np.cos(t)[:, None], np.sin(t)[:, None]
    x, y = pts[..., 0].copy(), pts[..., 1].copy()
    pts[..., 0] = c * x - s * y
    pts[..., 1] = s * x + c * y
    return pts.reshape(len(pts), -1)


def augment_rotations(X: np.ndarray, y: np.ndarray, copies: int = 3, max_degrees: float = 30.0,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return the original samples plus ``copies`` randomly rotated versions of each."""
    rng = np.random.default_rng(seed)
    Xs = [np.asarray(X)] + [rotate_xy(X, rng.uniform(-max_degrees, max_degrees, len(X))) for _ in range(copies)]
    return np.concatenate(Xs), np.tile(np.asarray(y), copies + 1)


def to_isotropic(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Undo MediaPipe's per-axis normalisation so x, y, z share one unit.

    x is normalised by width and y by height, so on a non-square frame
    (e.g. a 640x480 webcam) the raw values distort the hand shape. MediaPipe's
    z uses roughly the same scale as x.
    """
    lm = np.asarray(landmarks, dtype=np.float64).reshape(-1, 21, 3).copy()
    lm[..., 0] *= width
    lm[..., 1] *= height
    lm[..., 2] *= width
    return lm / max(width, height)


def normalize_landmarks(landmarks: np.ndarray, rotate: bool = False) -> np.ndarray:
    """Make landmarks invariant to hand position and size (and optionally rotation).

    Accepts ``(21, 3)`` or ``(N, 21, 3)`` and returns the same shape.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    single = lm.ndim == 2
    lm = lm.reshape(-1, 21, 3)

    pts = lm - lm[:, WRIST : WRIST + 1, :]  # translation
    palm = np.linalg.norm(pts[:, MIDDLE_MCP, :], axis=1)  # scale reference
    pts = pts / np.maximum(palm, 1e-9)[:, None, None]

    if rotate:
        # Rotate in the image plane so the wrist -> middle-MCP axis points up (-y).
        vx, vy = pts[:, MIDDLE_MCP, 0], pts[:, MIDDLE_MCP, 1]
        theta = np.arctan2(vx, -vy)  # angle of the palm axis from "up"
        c, s = np.cos(theta), np.sin(theta)
        x, y = pts[..., 0].copy(), pts[..., 1].copy()
        pts[..., 0] = c[:, None] * x + s[:, None] * y
        pts[..., 1] = -s[:, None] * x + c[:, None] * y

    return pts[0] if single else pts


def mirror_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Mirror horizontally (left hand -> right hand), in normalised image coordinates."""
    lm = np.array(landmarks, dtype=np.float64, copy=True)
    lm[..., 0] = 1.0 - lm[..., 0]
    return lm


def build_features(landmarks: np.ndarray, feature_set: str) -> np.ndarray:
    """Turn ``(N, 21, 3)`` (or a single ``(21, 3)``) landmarks into an ``(N, 63)`` matrix."""
    lm = np.asarray(landmarks, dtype=np.float64).reshape(-1, 21, 3)
    if feature_set == "raw":
        out = lm
    elif feature_set in ("norm", "norm_aug"):
        out = normalize_landmarks(lm, rotate=False)
    elif feature_set == "norm_rot":
        out = normalize_landmarks(lm, rotate=True)
    else:
        raise ValueError(f"Unknown feature set {feature_set!r}; choose from {FEATURE_SETS}")
    return out.reshape(len(lm), -1)
