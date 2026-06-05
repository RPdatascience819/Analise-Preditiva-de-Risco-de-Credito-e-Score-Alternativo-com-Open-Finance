"""Project paths and shared configuration."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_HOME_CREDIT_ZIP = RAW_DATA_DIR / "home-credit-default-risk.zip"
PROCESSED_TRAIN_DATASET = PROCESSED_DATA_DIR / "home_credit_train_enriched.parquet"

RANDOM_STATE = 42
TARGET_COLUMN = "TARGET"
