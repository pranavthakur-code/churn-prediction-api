import numpy as np
import pandas as pd
import pytest

from churn_core.drift import build_reference, compute_drift, psi
from churn_core.features import Cleaner


def test_psi_zero_for_identical_distributions():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0)


def test_psi_hand_computed():
    # (0.5-0.4)ln(0.5/0.4) + (0.5-0.6)ln(0.5/0.6)
    expected = 0.1 * np.log(1.25) + (-0.1) * np.log(0.5 / 0.6)
    assert psi([0.4, 0.6], [0.5, 0.5]) == pytest.approx(expected)


def test_psi_finite_when_a_bin_is_empty():
    assert np.isfinite(psi([0.5, 0.5], [1.0, 0.0]))


@pytest.fixture(scope="module")
def reference(raw_df):
    clean = Cleaner().transform(raw_df)
    return clean, build_reference(clean)


def test_resample_of_training_data_is_not_flagged(reference):
    clean, ref = reference
    res = compute_drift(ref, clean.sample(500, random_state=1))
    assert not res["drift_detected"], res["flagged_features"]


def test_shifted_inputs_are_flagged(reference):
    clean, ref = reference
    shifted = clean.sample(500, random_state=1).copy()
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] + 40
    shifted["Contract"] = "Month-to-month"
    res = compute_drift(ref, shifted)
    assert {"MonthlyCharges", "Contract"} <= set(res["flagged_features"])
    assert res["features"]["MonthlyCharges"]["status"] == "significant"


def test_unseen_category_counts_as_drift(reference):
    clean, ref = reference
    d = clean.sample(300, random_state=2).copy()
    d["PaymentMethod"] = "Crypto"
    assert "PaymentMethod" in compute_drift(ref, d)["flagged_features"]


def test_threshold_is_respected(reference):
    clean, ref = reference
    shifted = clean.sample(500, random_state=1).copy()
    shifted["tenure"] = (shifted["tenure"] * 0.7).round()
    assert compute_drift(ref, shifted, threshold=10)["drift_detected"] is False
