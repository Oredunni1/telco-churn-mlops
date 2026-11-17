"""Contrast test for compute metrics return shape, metrics bound and confusion matrics invariant"""

import pytest
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import LogisticRegression
from src import train, features

# Build the Pipeline
@pytest.fixture(scope="session")
def pipe():
    return Pipeline(
        [
            ("preprocessor", features.build_preprocessor()),
            ("classifier", LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)),
        ]
    )

def test_compute_metrics_return_shape(split_data, pipe):
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = split_data
    metric_keys = {
        "pipe", "cv_roc_auc_mean", "cv_roc_auc_std",
        "test_roc_auc", "test_precision", "test_recall", "test_f1",
        "confusion_matrix", "classification_report", "n_iterations",
    }
    result =  train.compute_metrics(pipe=pipe, X_train=X_TRAIN, X_test=X_TEST, y_train=Y_TRAIN, y_test=Y_TEST)
    assert set(result.keys()) == metric_keys

def test_metrics_boundaries(split_data, pipe):
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = split_data
    result  = train.compute_metrics(pipe=pipe, X_train=X_TRAIN, X_test=X_TEST, y_train=Y_TRAIN, y_test=Y_TEST)
    for key in ["test_roc_auc", "test_precision", "test_recall", "test_f1"]:
        val = result[key]
        assert isinstance(val, (float, np.floating)), (
            f"Metric {key} expected float type, got {type(val)}"
        )
        assert 0.0 <= val <= 1.0, (
            f"Metric {key} value {val} is out of bounds [0.0, 1.0]"
        )

def test_confusion_matrix_invariants(split_data, pipe):
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = split_data
    result = train.compute_metrics(pipe=pipe, X_train=X_TRAIN, X_test=X_TEST, y_train=Y_TRAIN, y_test=Y_TEST)
    cm = result["confusion_matrix"]
    assert cm.shape == (2, 2)
    assert cm.sum() == len(Y_TEST)
    assert (cm >= 0).all()
    
