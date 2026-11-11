import numpy as np
import pandas as pd
import pytest

from churn_core.features import CATEGORICAL, FEATURES, NUMERIC, Cleaner
from churn_core.pipeline import build_pipeline
from tests.conftest import CUSTOMER


def frame(**over):
    return pd.DataFrame([{**CUSTOMER, **over}])


def test_cleaner_turns_blank_total_charges_into_nan():
    out = Cleaner().transform(frame(TotalCharges=" "))
    assert np.isnan(out.loc[0, "TotalCharges"])


def test_cleaner_parses_numeric_text():
    out = Cleaner().transform(frame(TotalCharges="1889.50", tenure="34", MonthlyCharges="56.95"))
    assert out.loc[0, "TotalCharges"] == 1889.5 and out.loc[0, "tenure"] == 34.0
    assert all(out[c].dtype == "float64" for c in NUMERIC)


def test_cleaner_strips_whitespace_and_keeps_column_order():
    out = Cleaner().transform(frame(Contract="  One year "))
    assert out.loc[0, "Contract"] == "One year"
    assert list(out.columns) == FEATURES


def test_cleaner_reports_missing_columns():
    with pytest.raises(ValueError, match="missing required columns.*tenure"):
        Cleaner().transform(frame().drop(columns=["tenure"]))


def test_pipeline_handles_unseen_category_without_error(raw_df):
    pipe = build_pipeline("logistic")
    y = (raw_df["Churn"] == "Yes").astype(int)
    pipe.fit(raw_df.head(600), y.head(600))
    p = pipe.predict_proba(frame(Contract="Lifetime plan"))[:, 1]
    assert 0 <= p[0] <= 1


def test_pipeline_imputes_missing_numeric(raw_df):
    pipe = build_pipeline("logistic")
    y = (raw_df["Churn"] == "Yes").astype(int)
    pipe.fit(raw_df.head(600), y.head(600))
    assert np.isfinite(pipe.predict_proba(frame(TotalCharges=""))).all()


def test_unknown_model_family_rejected():
    with pytest.raises(ValueError):
        build_pipeline("svm")


def test_column_groups_do_not_overlap():
    assert not set(NUMERIC) & set(CATEGORICAL) and len(FEATURES) == 19
