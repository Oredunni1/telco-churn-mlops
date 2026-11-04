"""Train baseline churn model. Reports ROC-AUC, Precision, Recall, F1 via stratified CV, logs to MLflow.

Design Considerations:
1. Preprocess the dataset from the import of the feature, combined with the model instantiation in a pipeline 
2. class_weight='balanced' reweights each class by inverse frequency — with ~26.5% churn, the minority 'Yes' class gets more weight per example, so the model favors recall on churners at the cost of precision.
3. All MLflow logs are stated, including the log_param, log_metric and model artifact
"""

import mlflow
import mlflow.sklearn
import pandas as pd
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

def train_baseline():
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    mlflow.set_experiment("telco-churn")
    # Load and Split
    df = pd.read_parquet(data.DATA_PROCESS_FILE)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    # Build the Pipeline.
    pipe = Pipeline(
        [
            ("preprocessor", features.build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    # Aggregate CV scores
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Fit the model for training
    pipe.fit(X_train, y_train)
    n_iters = pipe.named_steps["classifier"].n_iter_[0]
    print(f"Converged in {n_iters} iterations")

    # Predictions
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    # Test Metrics
    roc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # --- MLFLOW TRACKING BLOCK ---
    with mlflow.start_run(run_name="logreg_balanced"):
        # Log Parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", pipe.named_steps['classifier'].class_weight)
        mlflow.log_param("max_iter", pipe.named_steps['classifier'].max_iter)
        mlflow.log_param("n_iterations", n_iters)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("cv_splits", CV_SPLITS)

        # Log Metrics
        mlflow.log_metric("cv_roc_auc_mean", cv_mean)
        mlflow.log_metric("cv_roc_auc_std", cv_std)
        mlflow.log_metric("test_roc_auc", roc)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)

        # Log Model Artifact
        mlflow.sklearn.log_model(pipe, name="model")
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")

    # Print summary report for visibility
    print("\n--- Cross-Validation Evaluation ---")
    print(f"CV ROC-AUC Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")

    print("\n--- Test Set Evaluation ---")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "roc_auc": roc, "precision": precision, "recall": recall, "f1": f1, "cm": cm,
    }


if __name__ == "__main__":
    train_baseline()