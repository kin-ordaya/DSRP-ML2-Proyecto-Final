"""Inference helpers for the best Telco Churn model.

This module exposes functions to load the persisted artifacts (preprocessor +
best model bundle) and to produce predictions with the tuned decision
threshold. It mirrors the exact transformations used during training so that
offline evaluation and future online scoring stay consistent.
"""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd

from src.features.build_features import load_preprocessor

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

TARGET_POSITIVE = "Yes"


def load_best_model() -> Dict:
    """Load the persisted best-model bundle.

    Returns:
        Dict: Mapping with keys ``model_name``, ``model``, ``threshold`` and
        ``metrics``.
    """
    return joblib.load(BEST_MODEL_PATH)


def predict_artifacts(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """Return the artifact bundle and a transformed feature matrix.

    The preprocessor is loaded from ``models/preprocessor.joblib`` and applied
    to the input dataframe.

    Args:
        df: Raw dataframe with the same columns as the training features.

    Returns:
        Tuple[Dict, pd.DataFrame]: The best-model bundle and the transformed
        features as a DataFrame.
    """
    preprocessor = load_preprocessor()
    X_transformed = preprocessor.transform(df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    return load_best_model(), X_transformed


def predict_proba(df: pd.DataFrame) -> pd.DataFrame:
    """Compute positive-class probabilities for each row.

    Args:
        df: Raw dataframe with the training feature columns.

    Returns:
        pd.DataFrame: Columns ``probability`` (positive-class probability) and
        ``prediction`` (binary label using the tuned threshold).
    """
    bundle, X = predict_artifacts(df)
    model = bundle["model"]
    threshold = bundle["threshold"]
    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "probability": proba,
            "prediction": (proba >= threshold).astype(int),
        }
    )


if __name__ == "__main__":
    from src.features.build_features import load_processed_data

    _, X_test, _, _ = load_processed_data()
    out = predict_proba(X_test)
    print(out.head(10))
    print(f"\nPredicción 'Yes' (1): {(out['prediction'] == 1).mean():.3f}")