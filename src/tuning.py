# src/tuning.py
"""Hyperparameter tuning for LogReg. GridSearchCV on X_train only, evaluate on X_test, log to MLflow."""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    
)
from sklearn.pipeline import Pipeline
from src import features, data

TEST_SIZE = 0.20
CV_SPLITS = 5
MLFLOW_DB_URI = f"sqlite:///{data.BASE_DIR}/mlflow.db"

def tune_logreg(X_train, y_train, X_test, y_test):
    # Build the Pipeline
    pipe = Pipeline(
        [
            ("preprocessor", features.build_preprocessor()),
            ('classifier', LogisticRegression(class_weight='balanced', random_state=42)),
        ]
    )
    # build param_grid
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
    }
    # StratifiedKFold
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
    
    # GridSearchCV over pipeline
    grid_search = GridSearchCV(
        estimator= pipe, 
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv
    )
    # fit the grid search cv
    grid_search.fit(X_train, y_train)
    # refit best_estimator_
    best_model = grid_search.best_estimator_
    # Aggregate CV scores
    cv_mean = grid_search.best_score_
    cv_std = grid_search.cv_results_["std_test_score"][grid_search.best_index_]
    # predict on test
    y_pred_proba = best_model.predict_proba(X_test)[:,1]
    y_pred = best_model.predict(X_test)

    # Validation Metrics
    # Test Metrics
    roc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    # log to MLflow as logreg_tuned
    with mlflow.start_run(run_name='logreg_tuned'):
        mlflow.log_param('model_type', type(best_model.named_steps['classifier']).__name__)
        for k,v in grid_search.best_params_.items():
            mlflow.log_param(k,v)
        mlflow.log_param('test_size', TEST_SIZE)
        mlflow.log_param('cv_splits', CV_SPLITS)
        mlflow.log_param('n_iterations', best_model.named_steps['classifier'].n_iter_[0])

         # Log Metrics
        mlflow.log_metric("cv_roc_auc_mean", cv_mean)
        mlflow.log_metric("cv_roc_auc_std", cv_std)
        mlflow.log_metric("test_roc_auc", roc)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)

        # Log Model Artifact & Text Reports
        mlflow.sklearn.log_model(
            best_model, 
            name="model",
        )
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(
            classification_report(y_test, y_pred), "classification_report.txt"
        )

        mlflow.log_dict(grid_search.cv_results_, "cv_results.json")

    # Print summary report for visibility
    print(f"CV ROC-AUC Mean: {cv_mean:.4f}")
    print(f"Test ROC-AUC:    {roc:.4f}")
    print(f"Test Precision:  {precision:.4f}")
    print(f"Test Recall:     {recall:.4f}")
    print(f"Test F1-Score:   {f1:.4f}\n")

    return {
        "roc_auc": roc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm,
    }



if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    mlflow.set_experiment("telco-churn")
    # load data, split (same random_state=42)
    df = pd.read_parquet(data.DATA_PROCESS_FILE)
    X = df.drop("Churn", axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    # call tune_logreg
    tune_logreg(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)