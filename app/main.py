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

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, status
 
from app.schemas import CustomerFeatures, PredictionResponse
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
 
@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict(features: CustomerFeatures) -> PredictionResponse:
    """Score a single customer record and return churn probability and binary prediction."""
    input_df = pd.DataFrame([features.model_dump()])
    churn_probability = float(get_model().predict_proba(input_df)[0, 1])
    churn_prediction = int(churn_probability >= THRESHOLD)
 
    return PredictionResponse(
        churn_probability=round(churn_probability, 4),
        churn_prediction=churn_prediction,
        threshold=THRESHOLD,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)