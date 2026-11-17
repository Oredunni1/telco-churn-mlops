"""Pytest configuration and shared fixtures for contract and unit tests."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.model_selection import train_test_split

from src import data

TEST_SIZE = 0.20

@pytest.fixture(scope="session")
def split_data(synthetic_csv):
    """Split the synthetic data once for the session, mirroring production's
    test_size and stratify config so tests exercise the same split contract."""
    df = data.load_data(synthetic_csv)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)

@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory) -> Path:
    """Generate a session-scoped 100-row synthetic Telco churn CSV fixture.

    Mimics the exact raw schema of the Telco dataset:
    - 21 columns including customerID and Churn as strings.
    - ~73/27 class balance for Churn ("No" / "Yes").
    - Explicit edge case: at least one row with a blank string (" ") in TotalCharges.
    """
    np.random.seed(42)
    n_rows = 100

    # ~73% No, ~27% Yes churn distribution
    churn_vals = np.random.choice(["No", "Yes"], size=n_rows, p=[0.73, 0.27])

    data = {
        "customerID": [f"{i:04d}-TEST" for i in range(n_rows)],
        "gender": np.random.choice(["Female", "Male"], size=n_rows),
        "SeniorCitizen": np.random.choice([0, 1], size=n_rows, p=[0.8, 0.2]),
        "Partner": np.random.choice(["Yes", "No"], size=n_rows),
        "Dependents": np.random.choice(["Yes", "No"], size=n_rows),
        "tenure": np.random.randint(1, 72, size=n_rows),
        "PhoneService": np.random.choice(["Yes", "No"], size=n_rows, p=[0.9, 0.1]),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], size=n_rows),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n_rows),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], size=n_rows),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], size=n_rows),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], size=n_rows),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], size=n_rows),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], size=n_rows),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], size=n_rows),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n_rows),
        "PaperlessBilling": np.random.choice(["Yes", "No"], size=n_rows),
        "PaymentMethod": np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            size=n_rows,
        ),
        "MonthlyCharges": np.random.uniform(20.0, 120.0, size=n_rows).round(2),
        "TotalCharges": [str(np.random.uniform(50.0, 5000.0)) for _ in range(n_rows)],
        "Churn": churn_vals,
    }

    df = pd.DataFrame(data)

    # Inject the whitespace blank string edge case in TotalCharges
    df.loc[0, "TotalCharges"] = " "

    # Write to tmp_path
    tmp_dir = tmp_path_factory.mktemp("synthetic")
    csv_path = tmp_dir / "telco_synthetic.csv"
    df.to_csv(csv_path, index=False)

    return csv_path