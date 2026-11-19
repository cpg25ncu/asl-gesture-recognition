"""Project-wide paths and constants.

Every module imports paths from here so the pipeline can be pointed at a
different dataset location with a single environment variable
(``ASL_DATA_DIR``) instead of editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("ASL_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"  # raw/A/*.jpg ... raw/J/*.jpg
PROCESSED_DIR = DATA_DIR / "processed"
LANDMARKS_CSV = PROCESSED_DIR / "landmarks.csv"
FAILURES_TXT = PROCESSED_DIR / "detection_failures.txt"
SPLIT_CSV = PROCESSED_DIR / "split.csv"
ROBUSTNESS_CACHE = PROCESSED_DIR / "robustness"

REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"
MODELS_DIR = ROOT / "models"
EXAMPLES_DIR = ROOT / "examples"
DEFAULT_MODEL = MODELS_DIR / "asl_classifier.joblib"

LABELS: list[str] = list("ABCDEFGHIJ")
N_LANDMARKS = 21
SEED = 2025
TEST_SIZE = 0.30
CV_FOLDS = 5


def ensure_dirs() -> None:
    """Create output directories if they do not exist yet."""
    for d in (PROCESSED_DIR, FIG_DIR, TABLE_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
