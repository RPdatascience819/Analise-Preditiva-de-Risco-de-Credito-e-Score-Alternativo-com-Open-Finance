"""Credit score transformation utilities."""

from __future__ import annotations

import pandas as pd


def probability_to_score(
    probability: float | pd.Series,
    min_score: int = 300,
    max_score: int = 850,
) -> float | pd.Series:
    """Convert default probability into a score where higher is lower risk."""
    return max_score - probability * (max_score - min_score)


def assign_risk_band(score: float) -> str:
    """Assign a simple risk band from a credit score."""
    if score >= 750:
        return "low"
    if score >= 650:
        return "medium"
    return "high"
