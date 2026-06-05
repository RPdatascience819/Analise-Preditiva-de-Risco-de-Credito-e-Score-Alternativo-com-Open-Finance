"""Build the supervised Home Credit modeling dataset."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_TRAIN_DATASET, RAW_HOME_CREDIT_ZIP
from src.features import (
    aggregate_bureau_history,
    aggregate_installments_payments,
    aggregate_previous_applications,
    criar_features_open_finance_simulado,
)


def read_csv_from_zip(zip_path: Path, member_name: str, **kwargs) -> pd.DataFrame:
    """Read one CSV member from the Home Credit zip file."""
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as file:
            return pd.read_csv(file, **kwargs)


def build_home_credit_training_dataset(raw_zip_path: Path = RAW_HOME_CREDIT_ZIP) -> pd.DataFrame:
    """Build the enriched supervised training dataset from official Home Credit files."""
    if not raw_zip_path.exists():
        raise FileNotFoundError(f"Raw Home Credit zip not found: {raw_zip_path}")

    application_train = read_csv_from_zip(raw_zip_path, "application_train.csv")
    bureau = read_csv_from_zip(
        raw_zip_path,
        "bureau.csv",
        usecols=[
            "SK_ID_CURR",
            "SK_ID_BUREAU",
            "CREDIT_ACTIVE",
            "DAYS_CREDIT",
            "CREDIT_DAY_OVERDUE",
            "DAYS_CREDIT_UPDATE",
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_OVERDUE",
            "AMT_ANNUITY",
        ],
    )
    previous_application = read_csv_from_zip(
        raw_zip_path,
        "previous_application.csv",
        usecols=[
            "SK_ID_CURR",
            "SK_ID_PREV",
            "NAME_CONTRACT_STATUS",
            "AMT_ANNUITY",
            "AMT_APPLICATION",
            "AMT_CREDIT",
            "CNT_PAYMENT",
            "DAYS_DECISION",
        ],
    )
    installments_payments = read_csv_from_zip(
        raw_zip_path,
        "installments_payments.csv",
        usecols=[
            "SK_ID_CURR",
            "SK_ID_PREV",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ],
    )

    dataset = application_train.merge(aggregate_bureau_history(bureau), on="SK_ID_CURR", how="left")
    dataset = dataset.merge(aggregate_previous_applications(previous_application), on="SK_ID_CURR", how="left")
    dataset = dataset.merge(aggregate_installments_payments(installments_payments), on="SK_ID_CURR", how="left")

    engineered_columns = dataset.columns.difference(application_train.columns)
    dataset[engineered_columns] = dataset[engineered_columns].fillna(0)

    return criar_features_open_finance_simulado(dataset)


def save_dataset(dataset: pd.DataFrame, output_path: Path = PROCESSED_TRAIN_DATASET) -> None:
    """Save the enriched dataset as parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-zip", type=Path, default=RAW_HOME_CREDIT_ZIP)
    parser.add_argument("--output", type=Path, default=PROCESSED_TRAIN_DATASET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_home_credit_training_dataset(args.raw_zip)
    save_dataset(dataset, args.output)
    print(f"Saved {len(dataset):,} rows and {len(dataset.columns):,} columns to {args.output}")


if __name__ == "__main__":
    main()
