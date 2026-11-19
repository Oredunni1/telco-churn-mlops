"""API contract tests for the FastAPI churn prediction service.

Design Note — Why importing app.main at collection/test time matters for CI:
With lazy loading (`get_model()`), importing `app.main` does not load the
MLflow model at module import time. Patching `app.main._model` in an `autouse`
fixture ensures `get_model()` immediately returns our deterministic stub without
ever touching MLflow or requiring `mlflow.db` on CI, keeping test collection
and execution fast and self-contained.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import CustomerFeatures, app


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """Pre-populate app.main._model with a deterministic stub so MLflow is never called."""
    stub = MagicMock()
    stub.predict_proba.return_value = np.array([[0.2166, 0.7834]])
    monkeypatch.setattr(main, "_model", stub)
    return stub


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["run_id"], str)


def test_predict_valid_input():
    payload = CustomerFeatures.model_config["json_schema_extra"]["examples"][0]
    client = TestClient(app)

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in {0, 1}


def test_predict_invalid_input_returns_422():
    client = TestClient(app)

    response = client.post("/predict", json={"gender": "female"})

    assert response.status_code == 422
    body = response.json()
    assert "detail" in body