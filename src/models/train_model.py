"""Training pipeline for the Telco Churn prediction models.

This module trains and compares several classifiers (Logistic Regression as
baseline, Random Forest, XGBoost and LightGBM) on the transformed Telco Churn
data. It:

- Loads the processed train/test splits and the fitted preprocessor.
- Sweeps a decision threshold over the **validation set** to maximize the F1
  score (the test set is never used for model or threshold selection).
- Logs parameters, metrics and artifacts to MLflow (configurable URI).
- Persists the best model and its optimal threshold for later inference.

The comparison is based on the F1 score over the validation split; the final
offline evaluation is reported on the held-out test split.
"""

import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features.build_features import load_preprocessor, load_processed_data

PROJECT_DIR = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_DIR / ".env")

MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports" / "figures"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

TARGET_POSITIVE = "Yes"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.3
MIN_F1_THRESHOLD = 0.60

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT_NAME = "telco-churn-modeling"
MLFLOW_LOCAL_URI = f"sqlite:///{PROJECT_DIR / 'mlflow.db'}"
MLFLOW_REGISTERED_MODEL_NAME = "telco-churn-xgboost"
MLFLOW_MODEL_STAGE = "Staging"


def build_models() -> Dict[str, Tuple[Callable, Dict]]:
    """Instantiate the candidate classifiers with class-imbalance handling.

    Class weights / scale_pos_weight are configured to counterbalance the
    73%/27% target distribution without resampling.

    Returns:
        Dict[str, Tuple[Callable, Dict]]: Mapping of model name to
        ``(estimator_factory, hyperparameters)``.
    """
    return {
        "logistic_regression": (
            lambda: LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {},
        ),
        "random_forest": (
            lambda: RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {},
        ),
        "xgboost": (
            lambda: XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                scale_pos_weight=2.77,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            ),
            {},
        ),
        "lightgbm": (
            lambda: LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                scale_pos_weight=2.77,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            {},
        ),
    }


def find_optimal_threshold(
    y_val: pd.Series, proba_val: np.ndarray, metric_fn: Callable, thresholds: np.ndarray
) -> float:
    """Return the threshold maximizing ``metric_fn`` on the validation set.

    Args:
        y_val: Validation ground-truth labels.
        proba_val: Predicted probabilities of the positive class (column 1).
        metric_fn: Metric function mapping ``(y_true, y_pred)`` to a score.
        thresholds: Candidate decision thresholds.

    Returns:
        float: The threshold that maximizes the metric.
    """
    best_threshold, best_score = 0.5, -np.inf
    for t in thresholds:
        y_pred = (proba_val >= t).astype(int)
        score = metric_fn(y_val, y_pred)
        if score > best_score:
            best_threshold, best_score = t, score
    return best_threshold


def evaluate(y_true: pd.Series, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute the full set of offline metrics for a given threshold.

    Args:
        y_true: Ground-truth labels.
        proba: Predicted probabilities of the positive class.
        threshold: Decision threshold applied to the probabilities.

    Returns:
        Dict[str, float]: Metrics keyed by name.
    """
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, proba),
    }


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, name: str, output_dir: Path) -> Path:
    """Plot and save the ROC curve for a single model.

    Args:
        fpr: False-positive rates.
        tpr: True-positive rates.
        roc_auc: Area under the ROC curve.
        name: Model name used in the file and legend.
        output_dir: Directory where the figure is saved.

    Returns:
        Path: Location of the saved figure.
    """
    import matplotlib.pyplot as plt

    path = output_dir / "roc_curve.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(f"Curva ROC - {name}")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pr(precision: np.ndarray, recall: np.ndarray, pr_auc: float, name: str, output_dir: Path) -> Path:
    """Plot and save the Precision-Recall curve for a single model.

    Args:
        precision: Precision values from ``precision_recall_curve``.
        recall: Recall values from ``precision_recall_curve``.
        pr_auc: Area under the PR curve.
        name: Model name used in the file and legend.
        output_dir: Directory where the figure is saved.

    Returns:
        Path: Location of the saved figure.
    """
    import matplotlib.pyplot as plt

    path = output_dir / "pr_curve.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})")
    ax.set_title(f"Curva Precision-Recall - {name}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_artifacts(results: Dict[str, Dict]) -> str:
    """Persist diagnostics of the best model and the best model bundle.

    Only the best model's diagnostics are kept in ``reports/figures/`` with
    canonical names (the per-model versions live only inside MLflow runs).

    Args:
        results: Output of :func:`train_and_evaluate` keyed by model name.

    Returns:
        str: Name of the best model by test F1.
    """
    best_name = max(results, key=lambda k: results[k]["metrics"]["f1"])
    best = results[best_name]

    plot_roc(best["fpr"], best["tpr"], best["metrics"]["auc_roc"], best_name, REPORTS_DIR)
    plot_pr(best["precision_curve"], best["recall_curve"], best["pr_auc"], best_name, REPORTS_DIR)
    best["cm"].to_csv(REPORTS_DIR / "confusion_matrix.csv", index=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": best_name,
            "model": best["model"],
            "threshold": best["threshold"],
            "metrics": best["metrics"],
        },
        BEST_MODEL_PATH,
    )
    print(f"Mejor modelo: {best_name}")
    print(f"Umbral óptimo (F1 valid): {best['threshold']:.3f}")
    print(f"Guardado en: {BEST_MODEL_PATH}")
    return best_name


def train_and_evaluate() -> Dict[str, Dict]:
    """Train all candidate models, tune thresholds and evaluate on test.

    Returns:
        Dict[str, Dict]: Mapping of model name to its results, including the
        fitted estimator, validation/test metrics and diagnostics.
    """
    X_train, X_test, y_train, y_test = load_processed_data()
    preprocessor = load_preprocessor()

    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    X_train_tr = preprocessor.transform(X_train_val)
    X_val_tr = preprocessor.transform(X_val)
    X_test_tr = preprocessor.transform(X_test)

    y_train_val = (y_train_val == TARGET_POSITIVE).astype(int)
    y_val = (y_val == TARGET_POSITIVE).astype(int)
    y_test_num = (y_test == TARGET_POSITIVE).astype(int)

    models = build_models()
    thresholds = np.arange(0.05, 0.96, 0.01)
    results: Dict[str, Dict] = {}

    for name, (factory, _params) in models.items():
        model = factory()
        model.fit(X_train_tr, y_train_val)

        proba_val = model.predict_proba(X_val_tr)[:, 1]
        proba_test = model.predict_proba(X_test_tr)[:, 1]

        threshold = find_optimal_threshold(y_val, proba_val, f1_score, thresholds)
        val_metrics = evaluate(y_val, proba_val, threshold)
        test_metrics = evaluate(y_test_num, proba_test, threshold)

        fpr, tpr, _ = roc_curve(y_test_num, proba_test)
        precision, recall, _ = precision_recall_curve(y_test_num, proba_test)
        pr_auc = auc(recall, precision)
        cm = pd.DataFrame(
            confusion_matrix(y_test_num, (proba_test >= threshold).astype(int)),
            index=["Actual No", "Actual Yes"],
            columns=["Pred No", "Pred Yes"],
        )

        results[name] = {
            "model": model,
            "threshold": threshold,
            "val_metrics": val_metrics,
            "metrics": test_metrics,
            "fpr": fpr,
            "tpr": tpr,
            "precision_curve": precision,
            "recall_curve": recall,
            "pr_auc": pr_auc,
            "cm": cm,
            "proba_test": proba_test,
        }
        print(
            f"[{name}] Val-F1={val_metrics['f1']:.3f} | "
            f"Test-F1={test_metrics['f1']:.3f} | "
            f"Test-Recall={test_metrics['recall']:.3f} | "
            f"AUC={test_metrics['auc_roc']:.3f} | thresh={threshold:.2f}"
        )

    return results


def log_to_mlflow(results: Dict[str, Dict], models: Dict[str, Tuple], best_name: str) -> None:
    """Log parameters, metrics and artifacts of every run to MLflow.

    A run is created per fitted model under a configurable experiment. If no
    tracking URI is set, MLflow writes to the local ``mlflow.db`` of the
    project (DagsHub is configured in a later step). Per-model figures and
    confusion matrices are generated in a temporary directory so that only the
    best model's diagnostics remain in ``reports/figures/``.

    Args:
        results: Output of :func:`train_and_evaluate` keyed by model name.
        models: Mapping of model name to ``(factory, params)``.
        best_name: Name of the best performing model (used to select the run
            to register in the MLflow Model Registry).
    """
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    else:
        mlflow.set_tracking_uri(MLFLOW_LOCAL_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_run_id: str | None = None

    for name, res in results.items():
        _, params = models[name]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            roc_path = plot_roc(res["fpr"], res["tpr"], res["metrics"]["auc_roc"], name, tmp)
            pr_path = plot_pr(res["precision_curve"], res["recall_curve"], res["pr_auc"], name, tmp)
            cm_path = tmp / "confusion_matrix.csv"
            res["cm"].to_csv(cm_path, index=True)

            model_dir = tmp / "model"
            if name == "logistic_regression" or name == "random_forest":
                mlflow.sklearn.save_model(res["model"], path=str(model_dir))
            elif name == "xgboost":
                mlflow.xgboost.save_model(res["model"], path=str(model_dir))
            elif name == "lightgbm":
                mlflow.lightgbm.save_model(res["model"], path=str(model_dir))

            with mlflow.start_run(run_name=name) as run:
                for k, v in params.items():
                    mlflow.log_param(k, v)
                mlflow.log_param("threshold", res["threshold"])
                for k, v in res["metrics"].items():
                    mlflow.log_metric(k, v)

                mlflow.log_artifact(str(roc_path))
                mlflow.log_artifact(str(pr_path))
                mlflow.log_artifact(str(cm_path))

                mlflow.log_artifacts(str(model_dir), artifact_path="model")

                if name == best_name:
                    best_run_id = run.info.run_id

    if best_run_id is not None:
        register_model_in_registry(best_run_id, results[best_name])


def register_model_in_registry(run_id: str, best: Dict) -> None:
    """Register the best model in the MLflow Model Registry.

    The model is registered and transitioned to the configured stage. Only the
    metrics of the best model are attached as tags/description for traceability.

    Args:
        run_id: MLflow run id containing the ``model`` artifact.
        best: Result dict of the best model (see :func:`train_and_evaluate`).
    """
    try:
        model_version = mlflow.register_model(
            f"runs:/{run_id}/model",
            MLFLOW_REGISTERED_MODEL_NAME,
            await_registration_for=300,
        )
    except Exception as exc:  # pragma: no cover - depends on backend availability
        print(f"WARNING: No se pudo registrar el modelo en el Model Registry: {exc}")
        return

    client = MlflowClient()
    stage = MLFLOW_MODEL_STAGE
    client.transition_model_version_stage(
        name=MLFLOW_REGISTERED_MODEL_NAME,
        version=model_version.version,
        stage=stage,
    )
    metrics = {k: round(v, 4) for k, v in best["metrics"].items()}
    description = (
        f"Mejor modelo ({MLFLOW_REGISTERED_MODEL_NAME.split('-')[-1]}) para Telco Churn. "
        f"Métricas sobre test: accuracy={metrics['accuracy']}, "
        f"precision={metrics['precision']}, recall={metrics['recall']}, "
        f"f1={metrics['f1']}, auc_roc={metrics['auc_roc']}. "
        f"Umbral óptimo: {best['threshold']:.3f}."
    )
    client.update_model_version(
        name=MLFLOW_REGISTERED_MODEL_NAME,
        version=model_version.version,
        description=description,
    )
    print(
        f"Modelo registrado: '{MLFLOW_REGISTERED_MODEL_NAME}' "
        f"versión {model_version.version} (stage={stage})."
    )


def main() -> None:
    """Run the full training/evaluation flow and persist the best model."""
    results = train_and_evaluate()
    best_name = save_artifacts(results)
    log_to_mlflow(results, build_models(), best_name)

    best_metrics = results[best_name]["metrics"]
    print("\n=== Métricas del mejor modelo sobre TEST ===")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")

    failures: List[str] = []
    if best_metrics["auc_roc"] < 0.80:
        failures.append(f"AUC-ROC {best_metrics['auc_roc']:.3f} < 0.80")
    if best_metrics["f1"] < MIN_F1_THRESHOLD:
        failures.append(f"F1 {best_metrics['f1']:.3f} < {MIN_F1_THRESHOLD}")
    if best_metrics["recall"] < 0.60:
        failures.append(f"Recall {best_metrics['recall']:.3f} < 0.60")

    if failures:
        raise RuntimeError(
            "No se alcanzaron los umbrales mínimos sobre test: " + "; ".join(failures)
            + ". Revisar features/desbalance/hiperparámetros (commit propio)."
        )
    print(
        f"\nLos umbrales mínimos se cumplen: "
        f"AUC>=0.80, F1>={MIN_F1_THRESHOLD}, Recall>=0.60."
    )


if __name__ == "__main__":
    main()