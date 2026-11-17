"""Train and compare baseline churn models (LR / RF / GB). Reports ROC-AUC, Precision, Recall, F1 via stratified CV, logs to MLflow."""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import data, features

TEST_SIZE = 0.20
CV_SPLITS = 5
MLFLOW_DB_URI = f"sqlite:///{data.BASE_DIR}/mlflow.db"


def compute_metrics(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Compute cross-validation scores, fit the pipeline, and evaluate test metrics.

    Completely decoupled from MLflow for pure testability.
    """
    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Fit the model for training
    pipe.fit(X_train, y_train)

    # Safe iteration extraction (only LogisticRegression has n_iter_)
    trained_estimator = pipe.named_steps["classifier"]
    n_iters = (
        trained_estimator.n_iter_[0]
        if hasattr(trained_estimator, "n_iter_")
        else None
    )

    # Predictions
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    # Test Metrics
    roc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    return {
        "pipe": pipe,
        "cv_roc_auc_mean": cv_mean,
        "cv_roc_auc_std": cv_std,
        "test_roc_auc": roc,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "confusion_matrix": cm,
        "classification_report": class_report,
        "n_iterations": n_iters,
    }


def train_model(
    run_name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """Build the pipeline, compute metrics via compute_metrics, and log to MLflow."""
    # Build the Pipeline
    pipe = Pipeline(
        [
            ("preprocessor", features.build_preprocessor()),
            ("classifier", estimator),
        ]
    )

    # Compute metrics & fit pipeline (pure ML logic)
    metrics = compute_metrics(pipe, X_train, y_train, X_test, y_test)

    # --- MLFLOW TRACKING BLOCK ---
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag(
            "model_family",
            "linear" if isinstance(estimator, LogisticRegression) else "tree",
        )

        # Log Parameters safely
        mlflow.log_param("model_type", type(estimator).__name__)
        mlflow.log_param("n_estimators", getattr(estimator, "n_estimators", "N/A"))
        mlflow.log_param("class_weight", getattr(estimator, "class_weight", "N/A"))
        mlflow.log_param("max_iter", getattr(estimator, "max_iter", "N/A"))
        if metrics["n_iterations"] is not None:
            mlflow.log_param("n_iterations", metrics["n_iterations"])
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("cv_splits", CV_SPLITS)

        # Log Metrics
        mlflow.log_metric("cv_roc_auc_mean", metrics["cv_roc_auc_mean"])
        mlflow.log_metric("cv_roc_auc_std", metrics["cv_roc_auc_std"])
        mlflow.log_metric("test_roc_auc", metrics["test_roc_auc"])
        mlflow.log_metric("test_precision", metrics["test_precision"])
        mlflow.log_metric("test_recall", metrics["test_recall"])
        mlflow.log_metric("test_f1", metrics["test_f1"])

        # Log Model Artifact & Text Reports
        mlflow.sklearn.log_model(
            metrics["pipe"],
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.log_text(str(metrics["confusion_matrix"]), "confusion_matrix.txt")
        mlflow.log_text(metrics["classification_report"], "classification_report.txt")

    # Print summary report for visibility
    print(f"\n=== Results for {run_name} ===")
    print(f"CV ROC-AUC Mean: {metrics['cv_roc_auc_mean']:.4f} (+/- {metrics['cv_roc_auc_std']:.4f})")
    print(f"Test ROC-AUC:    {metrics['test_roc_auc']:.4f}")
    print(f"Test Precision:  {metrics['test_precision']:.4f}")
    print(f"Test Recall:     {metrics['test_recall']:.4f}")
    print(f"Test F1-Score:   {metrics['test_f1']:.4f}\n")


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    mlflow.set_experiment("telco-churn")

    # Load and Split
    df = pd.read_parquet(data.DATA_PROCESS_FILE)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    models = [
        (
            "logreg_balanced",
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(class_weight="balanced", random_state=42),
        ),
        ("gradient_boost", GradientBoostingClassifier(random_state=42)),
    ]

    for name, est in models:
        train_model(name, est, X_train, y_train, X_test, y_test)