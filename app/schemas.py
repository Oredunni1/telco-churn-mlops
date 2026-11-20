"""Pydantic request and response schemas for the churn prediction service."""

from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Input schema for customer churn prediction (19 raw features)."""

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


class PredictionResponse(BaseModel):
    """Output schema for customer churn predictions."""

    churn_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted probability of customer churn",
    )
    churn_prediction: Literal[0, 1] = Field(
        description="Binary churn decision after applying the classification threshold"
    )
    threshold: float = Field(
        gt=0.0,
        lt=1.0,
        description="Probability threshold used to produce churn_prediction",
    )