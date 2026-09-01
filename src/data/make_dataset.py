"""Utilities for downloading and saving the Telco Churn dataset.

This module handles the retrieval of the IBM Telco Customer Churn dataset
from Kaggle and its storage in ``data/raw/`` for use in the pipeline.
"""

from pathlib import Path

import kagglehub
import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATASET_KAGGLE_ID = "blastchar/telco-customer-churn"
DATASET_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_FILENAME = "telco_customer_churn.csv"


def download_dataset() -> Path:
    """Download the raw Telco Churn dataset from Kaggle.

    Returns:
        Path: Local path to the downloaded CSV file.
    """
    cache_dir = kagglehub.dataset_download(DATASET_KAGGLE_ID)
    return Path(cache_dir) / DATASET_FILE


def save_raw_dataset(destination: Path | None = None) -> Path:
    """Download the dataset and copy it into ``data/raw/``.

    Args:
        destination: Output file path. Defaults to ``data/raw/telco_customer_churn.csv``.

    Returns:
        Path: Final location of the raw CSV file.
    """
    target = destination or (RAW_DATA_DIR / OUTPUT_FILENAME)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    source = download_dataset()
    if source.resolve() != target.resolve():
        target.write_bytes(source.read_bytes())

    return target


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw Telco Churn dataset as a DataFrame.

    Args:
        path: Optional path to the CSV file. Defaults to ``data/raw/``.

    Returns:
        pd.DataFrame: The raw dataset content.
    """
    filepath = path or (RAW_DATA_DIR / OUTPUT_FILENAME)
    return pd.read_csv(filepath)


if __name__ == "__main__":
    saved_path = save_raw_dataset()
    print(f"Dataset guardado en: {saved_path}")