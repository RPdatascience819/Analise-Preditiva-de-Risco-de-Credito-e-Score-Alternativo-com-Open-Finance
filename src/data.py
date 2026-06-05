"""Data loading and saving helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Save a DataFrame as CSV, creating parent directories when needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, **kwargs)
