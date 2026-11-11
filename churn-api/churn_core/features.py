"""Column definitions and the cleaning step that is part of the model pipeline.

The Cleaner lives inside the scikit-learn Pipeline (and therefore inside the
pickled artefact). Whatever raw record training saw, serving sees too: there is
no separate "serving-time preprocessing" that could drift out of sync.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
# SeniorCitizen is a 0/1 flag in the source data; we treat it as a category
# so that it is handled like every other yes/no field.
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Churn"
ID_COLUMN = "customerID"


class Cleaner(BaseEstimator, TransformerMixin):
    """Select the model columns and repair the known data-quality problems.

    * TotalCharges is text in the source file; blank strings (new customers
      with tenure 0) become NaN and are imputed later in the pipeline.
    * Numeric columns are coerced to numbers (unparseable text -> NaN).
    * Categorical values are stripped of surrounding whitespace and turned
      into plain strings; missing values stay NaN for the imputer.
    """

    def fit(self, X, y=None):
        self._check_columns(X)
        return self

    @staticmethod
    def _check_columns(X):
        missing = [c for c in FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")

    def transform(self, X):
        self._check_columns(X)
        out = X[FEATURES].copy()
        for c in NUMERIC:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        for c in CATEGORICAL:
            out[c] = pd.Series(
                [np.nan if pd.isna(v) else str(v).strip() for v in out[c]],
                index=out.index, dtype=object)
            out.loc[out[c] == "", c] = np.nan
        return out
