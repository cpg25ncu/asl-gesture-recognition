"""Hand landmark detection with MediaPipe Hands.

``HandLandmarker`` wraps ``mediapipe.solutions.hands`` for two use cases:

* ``static_image_mode=True``  - every image is processed independently
  (dataset feature extraction).
* ``static_image_mode=False`` - video mode: the palm detector runs only when
  tracking is lost, and landmarks are tracked from frame to frame, which is
  what makes the real-time pipeline fast.

MediaPipe is imported lazily so that modules which only need the landmark
column names (and the unit tests) do not require it to be installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config

LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
COORD_COLUMNS = [f"{axis}{i}" for i in range(config.N_LANDMARKS) for axis in "xyz"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def imread(path: str | Path) -> np.ndarray | None:
    """Read an image as BGR. Unlike ``cv2.imread`` this works with non-ASCII paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: str | Path, image: np.ndarray) -> None:
    """Write an image; works with non-ASCII paths."""
    ok, buf = cv2.imencode(Path(path).suffix or ".png", image)
    if not ok:
        raise IOError(f"Could not encode image for {path}")
    buf.tofile(str(path))


@dataclass
class HandDetection:
    """One detected hand."""

    landmarks: np.ndarray  # (21, 3): x, y normalised to [0, 1] by width/height; z relative depth
    handedness: str  # MediaPipe label, which assumes a mirrored (selfie) image
    score: float  # handedness classification confidence
    raw: object = None  # original MediaPipe landmark proto, used for drawing


class HandLandmarker:
    """Thin, context-manager friendly wrapper around ``mp.solutions.hands.Hands``."""

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp  # lazy import

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, bgr: np.ndarray) -> HandDetection | None:
        """Return the first detected hand in a BGR image, or ``None``."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # lets MediaPipe avoid a copy
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        hand = result.multi_hand_landmarks[0]
        lm = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
        label, score = "Unknown", 0.0
        if result.multi_handedness:
            cls = result.multi_handedness[0].classification[0]
            label, score = cls.label, float(cls.score)
        return HandDetection(landmarks=lm, handedness=label, score=score, raw=hand)

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandLandmarker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def list_dataset_images(raw_dir: Path) -> list[tuple[str, Path]]:
    """Return ``(label, path)`` pairs for ``raw_dir/<label>/*.jpg``, sorted for reproducibility."""
    pairs: list[tuple[str, Path]] = []
    for label in config.LABELS:
        class_dir = raw_dir / label
        if not class_dir.is_dir():
            continue
        for p in class_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                pairs.append((label, p))
    pairs.sort(key=lambda t: (t[0], t[1].name))
    return pairs


def extract_dataset(
    raw_dir: Path = config.RAW_DIR,
    out_csv: Path = config.LANDMARKS_CSV,
    failures_txt: Path = config.FAILURES_TXT,
) -> dict:
    """Run MediaPipe over every image and write one row of 63 coordinates per image.

    Images where MediaPipe returns no hand are treated as noise: they are not
    written to the CSV but are listed (as paths relative to ``raw_dir``) in
    ``failures_txt`` so they can be inspected.
    """
    import pandas as pd

    pairs = list_dataset_images(raw_dir)
    if not pairs:
        raise FileNotFoundError(f"No images found under {raw_dir}/<A..J>/")

    rows, failures = [], []
    t0 = time.perf_counter()
    with HandLandmarker(static_image_mode=True) as detector:
        for i, (label, path) in enumerate(pairs, start=1):
            rel = path.relative_to(raw_dir).as_posix()
            img = imread(path)
            det = detector.detect(img) if img is not None else None
            if det is None:
                failures.append(rel)
            else:
                rows.append([rel, label, det.handedness, round(det.score, 4), *det.landmarks.reshape(-1).tolist()])
            if i % 500 == 0 or i == len(pairs):
                print(f"  processed {i}/{len(pairs)}  ok={len(rows)}  failed={len(failures)}")
    elapsed = time.perf_counter() - t0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["instance_id", "label", "handedness", "handedness_score", *COORD_COLUMNS])
    df.to_csv(out_csv, index=False)
    failures_txt.write_text("\n".join(failures) + "\n", encoding="utf-8")

    return {
        "n_images": len(pairs),
        "n_detected": len(rows),
        "n_failed": len(failures),
        "seconds": round(elapsed, 1),
        "images_per_second": round(len(pairs) / elapsed, 1),
    }
