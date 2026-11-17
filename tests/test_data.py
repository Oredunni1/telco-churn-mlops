"""Contract tests for data.load_data().

These tests verify the *interface* — that load_data returns a 7043-row DataFrame
with specific dtypes, no NaNs, and a valid target encoding. They do NOT test
that the model trains well on this data (see test_training.py); they test that
the data pipeline is a stable foundation for that training.
"""

import pytest
import pandas as pd
from pathlib import Path
from src import data


def test_load_data_contract_with_synthetic_csv(tmp_path):
    synthetic_csv = tmp_path / "synthetic.csv"
    
    # Construct minimal synthetic Telco data mirroring the raw CSV schema
    raw_data = {
        "customerID": ["1234-ABCD", "5678-EFGH"],
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "tenure": [1, 24],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "No"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges": ["29.85", " "],  # Edge case: blank string in TotalCharges
        "Churn": ["No", "Yes"]
    }
    pd.DataFrame(raw_data).to_csv(synthetic_csv, index=False)
    
    # Load via the actual function passing the synthetic path
    df = data.load_data(synthetic_csv)
    
    # Assertions on pipeline contract
    assert df.shape == (2, 20), f"Expected (2, 20), got {df.shape}"
    assert "customerID" not in df.columns, "customerID should be dropped"
    assert df["TotalCharges"].iloc[1] == 0.0, "Blank TotalCharges should be filled with 0.0"
    assert set(df["Churn"].unique()) == {0, 1}, "Churn should be binary encoded to 0 and 1"


# Load the cleaned dataset once for all tests in this module.
@pytest.fixture(scope="module")
def df(synthetic_csv):
    """Load the cleaned dataset once for all tests in this module."""
    return data.load_data(synthetic_csv)

def test_count_row(df, synthetic_csv):
    expected = len(pd.read_csv(synthetic_csv))
    assert len(df) == expected


def test_count_columns(df):
    assert len(df.columns) == 20, f"Expected 20 columns, got {len(df.columns)}"

def test_column_set(df):
    expected = {
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges", "Churn",
    }
    assert set(df.columns) == expected

def test_target_encoding(df):
    unique_vals = set(df["Churn"].unique())
    assert unique_vals == {0, 1}, f"Expected {{0, 1}}, got {unique_vals}"


def test_is_nan(df):
    null_count = df.isnull().sum().sum()
    assert null_count == 0, f"Found {null_count} missing values"


def test_dtype_check(df):
    assert pd.api.types.is_float_dtype(df["TotalCharges"]), (
        f"Expected float dtype for TotalCharges, got {df['TotalCharges'].dtype}"
    )

def test_churn_dtype(df):
    assert pd.api.types.is_integer_dtype(df["Churn"]), (
        f"Expected integer dtype for Churn, got {df['Churn'].dtype}"
    )

def test_churn_rate_plausible(df):
    rate = df["Churn"].mean()
    assert 0.15 <= rate <= 0.40, f"Churn rate {rate:.4f} outside plausible bounds"

