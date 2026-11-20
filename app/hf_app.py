"""Gradio UI wrapper for serving the churn model on Hugging Face Spaces.

Design Decision — `gr.Interface` vs. `gr.Blocks`:
We use `gr.Interface` rather than `gr.Blocks` because it provides a clean,
declarative form layout with minimal boilerplate for a single-prediction demo.
`gr.Blocks` is better suited when you need custom multi-column layouts,
interactive tabs, or side-by-side probability + explanation panels.
"""

from pathlib import Path
from typing import Literal

import gradio as gr
import joblib
import pandas as pd

from app.schemas import CustomerFeatures

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logreg_balanced.pkl"
THRESHOLD = 0.5

model = joblib.load(MODEL_PATH)
EXAMPLE = CustomerFeatures.model_config["json_schema_extra"]["examples"][0]


def predict_from_ui(
    gender: Literal["Female", "Male"],
    SeniorCitizen: int,
    Partner: Literal["Yes", "No"],
    Dependents: Literal["Yes", "No"],
    tenure: int,
    PhoneService: Literal["Yes", "No"],
    MultipleLines: Literal["Yes", "No", "No phone service"],
    InternetService: Literal["DSL", "Fiber optic", "No"],
    OnlineSecurity: Literal["Yes", "No", "No internet service"],
    OnlineBackup: Literal["Yes", "No", "No internet service"],
    DeviceProtection: Literal["Yes", "No", "No internet service"],
    TechSupport: Literal["Yes", "No", "No internet service"],
    StreamingTV: Literal["Yes", "No", "No internet service"],
    StreamingMovies: Literal["Yes", "No", "No internet service"],
    Contract: Literal["Month-to-month", "One year", "Two year"],
    PaperlessBilling: Literal["Yes", "No"],
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    MonthlyCharges: float,
    TotalCharges: float,
) -> str:
    """Validate inputs with CustomerFeatures, run inference, and format risk output."""
    record = CustomerFeatures(
        gender=gender,
        SeniorCitizen=int(SeniorCitizen),
        Partner=Partner,
        Dependents=Dependents,
        tenure=int(tenure),
        PhoneService=PhoneService,
        MultipleLines=MultipleLines,
        InternetService=InternetService,
        OnlineSecurity=OnlineSecurity,
        OnlineBackup=OnlineBackup,
        DeviceProtection=DeviceProtection,
        TechSupport=TechSupport,
        StreamingTV=StreamingTV,
        StreamingMovies=StreamingMovies,
        Contract=Contract,
        PaperlessBilling=PaperlessBilling,
        PaymentMethod=PaymentMethod,
        MonthlyCharges=float(MonthlyCharges),
        TotalCharges=float(TotalCharges),
    )

    df = pd.DataFrame([record.model_dump()])
    prob = float(model.predict_proba(df)[0, 1])
    risk_label = "HIGH RISK" if prob >= THRESHOLD else "LOW RISK"
    return f"Churn probability: {prob * 100:.1f}% → {risk_label}"


inputs = [
    gr.Dropdown(["Female", "Male"], value=EXAMPLE["gender"], label="gender"),
    gr.Dropdown([0, 1], value=EXAMPLE["SeniorCitizen"], label="SeniorCitizen"),
    gr.Dropdown(["Yes", "No"], value=EXAMPLE["Partner"], label="Partner"),
    gr.Dropdown(["Yes", "No"], value=EXAMPLE["Dependents"], label="Dependents"),
    gr.Dropdown(list(range(1, 73)), value=EXAMPLE["tenure"], label="tenure"),
    gr.Dropdown(["Yes", "No"], value=EXAMPLE["PhoneService"], label="PhoneService"),
    gr.Dropdown(
        ["Yes", "No", "No phone service"],
        value=EXAMPLE["MultipleLines"],
        label="MultipleLines",
    ),
    gr.Dropdown(
        ["DSL", "Fiber optic", "No"],
        value=EXAMPLE["InternetService"],
        label="InternetService",
    ),
    gr.Dropdown(
        ["Yes", "No", "No internet service"],
        value=EXAMPLE["OnlineSecurity"],
        label="OnlineSecurity",
    ),
    gr.Dropdown(
        ["Yes", "No", "No internet service"],
        value=EXAMPLE["OnlineBackup"],
        label="OnlineBackup",
    ),
    gr.Dropdown(
        ["Yes", "No", "No internet service"],
        value=EXAMPLE["DeviceProtection"],
        label="DeviceProtection",
    ),
    gr.Dropdown(
        ["Yes", "No", "No internet service"],
        value=EXAMPLE["TechSupport"],
        label="TechSupport",
    ),
    gr.Dropdown(
        ["Yes", "No", "No internet service"],
        value=EXAMPLE["StreamingTV"],
        label="StreamingTV",
    ),
    gr.Dropdown(
        ["Yes", "No", "No internet service"],
        value=EXAMPLE["StreamingMovies"],
        label="StreamingMovies",
    ),
    gr.Dropdown(
        ["Month-to-month", "One year", "Two year"],
        value=EXAMPLE["Contract"],
        label="Contract",
    ),
    gr.Dropdown(
        ["Yes", "No"],
        value=EXAMPLE["PaperlessBilling"],
        label="PaperlessBilling",
    ),
    gr.Dropdown(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        value=EXAMPLE["PaymentMethod"],
        label="PaymentMethod",
    ),
    gr.Number(value=EXAMPLE["MonthlyCharges"], label="MonthlyCharges"),
    gr.Number(value=EXAMPLE["TotalCharges"], label="TotalCharges"),
]

demo = gr.Interface(
    fn=predict_from_ui,
    inputs=inputs,
    outputs="text",
    title="Telco Customer Churn Predictor",
    description="Enter customer attributes to estimate churn probability and risk tier.",
)

if __name__ == "__main__":
    demo.launch()