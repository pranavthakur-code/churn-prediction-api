"""Request and response models. Pydantic rejects invalid input before the model sees it."""
from __future__ import annotations

import math
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Cat = Annotated[str, Field(min_length=1, max_length=64,
                           description="Category label; unseen labels are accepted with a warning")]


class Customer(BaseModel):
    """One customer record, using the column names of the public Telco churn dataset."""
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True,
        json_schema_extra={"example": {
            "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0,
            "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No",
            "MultipleLines": "No phone service", "InternetService": "DSL",
            "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
            "TotalCharges": 29.85}})

    customerID: Optional[str] = Field(None, max_length=64,
                                      description="Echoed back; not used by the model")
    gender: Cat
    SeniorCitizen: int = Field(ge=0, le=1, description="1 if senior citizen, else 0")
    Partner: Cat
    Dependents: Cat
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: Cat
    MultipleLines: Cat
    InternetService: Cat
    OnlineSecurity: Cat
    OnlineBackup: Cat
    DeviceProtection: Cat
    TechSupport: Cat
    StreamingTV: Cat
    StreamingMovies: Cat
    Contract: Cat
    PaperlessBilling: Cat
    PaymentMethod: Cat
    MonthlyCharges: float = Field(ge=0, le=1000, description="Current monthly bill, USD")
    TotalCharges: Optional[float] = Field(
        None, ge=0, le=200000,
        description="Lifetime billed amount, USD. Null or empty for brand-new customers "
                    "(the source data stores this column as text)")

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def blank_total_charges(cls, v):
        # the source system sends numbers as text and uses blanks for "no bill yet"
        return None if isinstance(v, str) and v.strip() == "" else v

    @field_validator("MonthlyCharges", "TotalCharges")
    @classmethod
    def finite(cls, v):
        if v is not None and not math.isfinite(v):
            raise ValueError("must be a finite number")
        return v


class Prediction(BaseModel):
    customer_id: Optional[str] = Field(None, description="customerID from the request, if given")
    churn_probability: float = Field(ge=0, le=1)
    churn_predicted: bool = Field(description="True when probability >= the model's threshold")
    threshold: float
    warnings: list[str] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    prediction: Prediction
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[Customer] = Field(min_length=1, description="1 to CHURN_MAX_BATCH records")


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    count: int
    model_version: str


class FeatureDrift(BaseModel):
    psi: float
    status: str = Field(description="stable (<0.1), moderate (0.1-0.2) or significant (>0.2)")
    flagged: bool


class DriftResponse(BaseModel):
    model_version: str
    n_rows: int
    threshold: float
    drift_detected: bool
    flagged_features: list[str]
    features: dict[str, FeatureDrift]


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ModelInfo(BaseModel):
    model_version: str
    model_family: str
    threshold: float
    trained_at: str
    training_data_sha256: str
    model_sha256: str
    cost_model: dict
    test_metrics: dict
    libraries: dict


class ErrorItem(BaseModel):
    field: str
    message: str
    type: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: Optional[str] = None
    errors: list[ErrorItem] = Field(default_factory=list)
