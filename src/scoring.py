"""Credit score transformation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def probability_to_score(
    probability: float | pd.Series,
    min_score: int = 300,
    max_score: int = 850,
) -> float | pd.Series:
    """Convert default probability into a score where higher is lower risk."""
    return max_score - probability * (max_score - min_score)


def probability_to_scorecard_points(
    probability,
    pdo: int = 20,
    odds_ref: float = 50.0,
    score_ref: int = 600,
    min_score: int = 300,
    max_score: int = 850,
):
    """Convert default probability into scorecard points (industry PDO method).

    Uses the log-odds scaling Score = offset + factor * ln(odds), where
    odds = good/bad = (1 - p) / p, factor = pdo / ln(2) and
    offset = score_ref - factor * ln(odds_ref). Higher points = lower risk,
    and every ``pdo`` extra points doubles the good/bad odds.

    Accepts a scalar, a numpy array or a pandas Series and returns integer
    points clipped to ``[min_score, max_score]`` (Series input preserves index).
    """
    factor = pdo / np.log(2)
    offset = score_ref - factor * np.log(odds_ref)

    prob = np.clip(probability, 0.001, 0.999)  # avoid log(0)/log(inf)
    odds = (1 - prob) / prob
    points = np.clip(offset + factor * np.log(odds), min_score, max_score)
    points = np.round(points).astype(int)

    if isinstance(probability, pd.Series):
        return pd.Series(points, index=probability.index)
    return points


def adjust_probability_to_prior(probability, target_prior: float, source_prior: float = 0.5):
    """Re-scale probabilities from a model trained under ``source_prior`` back to
    the true population prior ``target_prior``.

    Models trained on rebalanced data (e.g. SMOTE 50/50) output probabilities
    biased toward the training prior. This corrects the intercept shift on the
    log-odds: ``logit_true = logit_model + logit(target) - logit(source)``,
    preserving the ranking. Returns probabilities on the true scale (Series
    input preserves index).
    """
    eps = 1e-9
    p = np.clip(probability, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    adj = np.log(target_prior / (1 - target_prior)) - np.log(source_prior / (1 - source_prior))
    corrected = 1 / (1 + np.exp(-(logit + adj)))
    if isinstance(probability, pd.Series):
        return pd.Series(corrected, index=probability.index)
    return corrected


def assign_risk_band(score: float, medium_min: float = 650, low_min: float = 750) -> str:
    """Assign a risk band from a credit score.

    Thresholds are configurable so bands can be calibrated to the actual score
    distribution (e.g. tercile cut-offs) instead of fixed absolute values.
    """
    if score >= low_min:
        return "low"
    if score >= medium_min:
        return "medium"
    return "high"
