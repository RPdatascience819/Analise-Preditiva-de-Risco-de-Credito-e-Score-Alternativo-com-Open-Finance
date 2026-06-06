"""Model training utilities."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET_COLUMN


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> tuple[pd.DataFrame, pd.Series]:
    """Split a modelable dataset into features and target."""
    return df.drop(columns=[target_col]), df[target_col]


def train_baseline_classifier(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train a baseline classifier and return the fitted model plus train/test split."""
    X, y = split_features_target(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    model = RandomForestClassifier(
        random_state=random_state,
        min_samples_leaf=1,
        max_features="sqrt",
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def save_model(model, path: str | Path) -> None:
    """Persist a trained model with joblib."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
