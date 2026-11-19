"""FastAPI service serving the churn model from MLflow.

Design Decisions:
- Q1 (Startup load vs. per-request load): The model is cached lazily via
  `get_model()` on first use (and warmed during app lifespan startup) rather than
  loaded on every request, avoiding ~50–200ms per-call reload overhead while
  keeping module imports cheap and CI-friendly.
- Q2 (Model load failure): Crash fast so uvicorn fails startup with a clear
  traceback, because a service that starts up only to silently fail on every
  /predict call is worse than one that refuses to start in a broken state.
- Q3 (Returning both probability and label): Returning both gives callers a
  ready-to-use thresholded decision while letting them rank customers by risk
  or apply custom thresholds downstream without redeploying.
- Sync endpoint execution: /predict uses `def` so FastAPI runs synchronous
  CPU-bound inference in a threadpool, keeping the event loop free.
"""

from contextlib import asynccontextmanager
import os
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, status
from pydantic import BaseModel, Field

from src import data

RUN_ID = os.getenv("MODEL_RUN_ID", "b1d20639f6fd4e6c83da0053f9a538ac")
THRESHOLD = 0.5
MLFLOW_DB_URI = f"sqlite:///{data.BASE_DIR}/mlflow.db"

_model = None


def get_model():
    """Lazily load and cache the MLflow sklearn pipeline."""
    global _model
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_DB_URI)
        _model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at server startup so broken configurations fail fast."""
    get_model()
    yield


class CustomerFeatures(BaseModel):
    """Input schema for customer churn prediction."""

    gender: Literal["Female", "Male"] = Field(
        description="Customer gender ('Female' or 'Male')"
    )
    SeniorCitizen: Literal[0, 1] = Field(
        description="Whether the customer is a senior citizen (0 = No, 1 = Yes)"
    )
    Partner: Literal["Yes", "No"] = Field(
        description="Whether the customer has a partner ('Yes' or 'No')"
    )
    Dependents: Literal["Yes", "No"] = Field(
        description="Whether the customer has dependents ('Yes' or 'No')"
    )
    tenure: int = Field(
        ge=1,
        le=72,
        description="Number of months the customer has stayed with the company (1 to 72)",
    )
    PhoneService: Literal["Yes", "No"] = Field(
        description="Whether the customer has phone service ('Yes' or 'No')"
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        description="Whether the customer has multiple lines ('Yes', 'No', or 'No phone service')"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        description="Customer's internet service provider ('DSL', 'Fiber optic', or 'No')"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        description="Whether the customer has online security ('Yes', 'No', or 'No internet service')"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        description="Whether the customer has online backup ('Yes', 'No', or 'No internet service')"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        description="Whether the customer has device protection ('Yes', 'No', or 'No internet service')"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        description="Whether the customer has tech support ('Yes', 'No', or 'No internet service')"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        description="Whether the customer has streaming TV ('Yes', 'No', or 'No internet service')"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        description="Whether the customer has streaming movies ('Yes', 'No', or 'No internet service')"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        description="The contract term of the customer ('Month-to-month', 'One year', or 'Two year')"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        description="Whether the customer has paperless billing ('Yes' or 'No')"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        description=(
            "The customer's payment method ('Electronic check', 'Mailed check', "
            "'Bank transfer (automatic)', or 'Credit card (automatic)')"
        )
    )
    MonthlyCharges: float = Field(
        ge=0.0,
        description="The amount charged to the customer monthly",
    )
    TotalCharges: float = Field(
        ge=0.0,
        description="The total amount charged to the customer",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 1,
                    "PhoneService": "No",
                    "MultipleLines": "No phone service",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": 29.85,
                }
            ]
        }
    }


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Serve customer churn predictions from a trained MLflow model.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Verify the service is running and report the configured model run ID."""
    return {
        "status": "ok",
        "run_id": RUN_ID,
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(features: CustomerFeatures):
    """Score a single customer record and return churn probability and binary prediction."""
    input_df = pd.DataFrame([features.model_dump()])
    churn_probability = float(get_model().predict_proba(input_df)[0, 1])
    churn_prediction = int(churn_probability >= THRESHOLD)

    return {
        "churn_probability": round(churn_probability, 4),
        "churn_prediction": churn_prediction,
        "threshold": THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)