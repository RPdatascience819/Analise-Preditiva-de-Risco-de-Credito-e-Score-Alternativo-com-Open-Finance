"""Feature engineering utilities for the credit risk project."""

from __future__ import annotations

import pandas as pd


def add_income_expense_ratio(
    df: pd.DataFrame,
    income_col: str = "income",
    expense_col: str = "expenses",
    output_col: str = "expense_to_income_ratio",
) -> pd.DataFrame:
    """Create an expense-to-income ratio feature when source columns exist."""
    featured = df.copy()
    if income_col in featured.columns and expense_col in featured.columns:
        featured[output_col] = featured[expense_col] / featured[income_col].replace(0, pd.NA)
    return featured
