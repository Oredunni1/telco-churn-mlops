"""Contract tests for features.build_preprocessor() and feature definitions."""

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import pytest
import numpy as np

from src import features


def test_return_type():
    preprocessor = features.build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer), (
        f"Expected ColumnTransformer, got {type(preprocessor)}"
    )


def test_unfitted_by_default():
    preprocessor = features.build_preprocessor()
    # An unfitted preprocessor must raise NotFittedError when checked
    with pytest.raises(NotFittedError):
        check_is_fitted(preprocessor)


def test_columns_count():
    total = len(features.NUM_COLS) + len(features.CAT_COLS)
    assert total == 19, f"Expected 19 total features, got {total}"


def test_seniorcitizen_categorical():
    # Ensure exact string match with your features.py definition (watch out for trailing spaces)
    assert "SeniorCitizen" in features.CAT_COLS, (
        "SeniorCitizen should be classified as a categorical feature"
    )

def test_column_coverage(split_data):
    X_train, _, _, _ = split_data
    feature_cols = set(X_train.columns)
    declared = set(features.NUM_COLS) | set(features.CAT_COLS)
    assert declared == feature_cols, (
        f"Declared features {declared - feature_cols} not in data; "
        f"data columns {feature_cols - declared} not declared"
    )

def test_fit_transform_no_nan(split_data):
    X_train, _, y_train, _ = split_data
    preprocessor = features.build_preprocessor()
    X_transformed = preprocessor.fit_transform(X_train, y_train)
    assert X_transformed.shape[0] == len(X_train)
    assert not np.isnan(X_transformed).any(), "Preprocessor produced NaN — check imputation/encoding"