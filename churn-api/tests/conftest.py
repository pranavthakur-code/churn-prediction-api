import json
import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings
from training.train import run

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Telco-Customer-Churn.csv"
API_KEY = "test-key-123"
HEADERS = {"X-API-Key": API_KEY}

CUSTOMER = {
    "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL", "OnlineSecurity": "No",
    "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}


@pytest.fixture(scope="session")
def raw_df():
    return pd.read_csv(DATA, dtype=str, keep_default_na=False)


@pytest.fixture(scope="session")
def small_data(tmp_path_factory, raw_df):
    """A 1,500-row sample so the integration tests train in seconds."""
    p = tmp_path_factory.mktemp("data") / "small.csv"
    raw_df.sample(1500, random_state=0).to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory, small_data):
    root = tmp_path_factory.mktemp("artifacts")
    return run(small_data, root, fast=True)


@pytest.fixture()
def settings(artifact_dir):
    return Settings(model_dir=artifact_dir, api_keys=(API_KEY,), max_batch_size=50,
                    drift_min_rows=50)


@pytest.fixture()
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture()
def customer():
    return dict(CUSTOMER)


@pytest.fixture(scope="session")
def meta(artifact_dir):
    return json.loads((artifact_dir / "metadata.json").read_text())
