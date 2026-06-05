"""Model evaluation utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def classification_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute core binary classification metrics."""
    predictions = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y, probabilities)
    return metrics
