"""Feature engineering and data preparation for the Telco Churn pipeline.

This module transforms the raw Telco Churn dataset into a model-ready format:
it cleans the raw data, splits it into train/test sets (stratified), fits a
``ColumnTransformer`` preprocessing pipeline exclusively on the training split
to avoid data leakage, and persists both the processed splits and the fitted
preprocessor.
"""

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.make_dataset import load_raw_data

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [col for col in FEATURE_COLUMNS if col not in NUMERIC_FEATURES]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning to the raw dataset.

    Converts ``TotalCharges`` to numeric (blank values become NaN to be imputed
    later) and drops the ``customerID`` column which has no predictive value.

    Args:
        df: Raw dataframe from ``data/raw``.

    Returns:
        pd.DataFrame: Cleaned dataframe with model-ready columns.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.drop(columns=["customerID"])


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer for numeric and categorical features.

    - Numeric: median imputation followed by standard scaling.
    - Categorical: most-frequent imputation followed by one-hot encoding.

    Returns:
        ColumnTransformer: Unfitted preprocessing pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the cleaned data into stratified train and test sets.

    Args:
        df: Cleaned dataframe including the target column.

    Returns:
        Tuple with ``(X_train, X_test, y_train, y_test)``.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Persist the cleaned train/test splits as CSV files.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)


def save_preprocessor(preprocessor: ColumnTransformer) -> None:
    """Serialize the fitted preprocessor to ``models/``.

    Args:
        preprocessor: Fitted ColumnTransformer.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)


def load_preprocessor() -> ColumnTransformer:
    """Load the fitted preprocessor from ``models/``.

    Returns:
        ColumnTransformer: The previously fitted preprocessing pipeline.
    """
    return joblib.load(PREPROCESSOR_PATH)


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the persisted train/test splits.

    Returns:
        Tuple with ``(X_train, X_test, y_train, y_test)``.
    """
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv")[TARGET_COLUMN]
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test


def preprocess_dataset() -> None:
    """End-to-end preprocessing: clean, split, fit and persist artifacts."""
    df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = make_train_test_split(df)

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    save_preprocessor(preprocessor)
    save_processed_data(X_train, X_test, y_train, y_test)
    print(f"Preprocesamiento completado. Preprocessor: {PREPROCESSOR_PATH}")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")


if __name__ == "__main__":
    preprocess_dataset()