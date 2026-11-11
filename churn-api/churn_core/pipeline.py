"""Model pipeline: cleaning + imputation + encoding + scaling + classifier."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_core.features import CATEGORICAL, NUMERIC, Cleaner

SEED = 42


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # unseen categories at serving time become all-zero rows, never an error
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC),
                              ("cat", categorical, CATEGORICAL)])


def build_pipeline(family: str = "logistic") -> Pipeline:
    if family == "logistic":
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
    elif family == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=SEED)
    else:
        raise ValueError(f"unknown model family: {family!r}")
    return Pipeline([("clean", Cleaner()), ("prep", build_preprocessor()),
                     ("clf", clf)])
