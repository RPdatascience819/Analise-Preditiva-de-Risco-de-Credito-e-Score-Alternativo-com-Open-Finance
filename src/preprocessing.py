"""Data cleaning utilities for the credit risk project."""

from __future__ import annotations

import pandas as pd


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with snake-case-like lowercase column names."""
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return cleaned


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without duplicated rows."""
    return df.drop_duplicates().copy()
