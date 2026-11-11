import numpy as np
import pytest

from churn_core.costs import CostModel


def test_analytic_threshold_formula():
    assert CostModel(300, 30, 0.5).analytic_threshold() == pytest.approx(0.2)
    assert CostModel(600, 30, 0.5).analytic_threshold() == pytest.approx(0.1)


def test_total_cost_hand_computed():
    cm = CostModel(customer_value=100, offer_cost=10, save_rate=0.5)
    y = [1, 1, 0, 0]
    p = [0.9, 0.1, 0.8, 0.1]
    # offered: #1 churner (10 + 0.5*100 = 60), #3 non-churner (10)
    # ignored: #2 churner (100), #4 non-churner (0)
    assert cm.total_cost(y, p, 0.5) == pytest.approx(170)


def test_baselines_hand_computed():
    cm = CostModel(100, 10, 0.5)
    y = [1, 1, 0, 0]
    b = cm.baselines(y)
    assert b["no_action"] == pytest.approx(200)
    assert b["offer_to_everyone"] == pytest.approx(4 * 10 + 2 * 50)


def test_best_threshold_on_perfect_scores_is_between_classes():
    cm = CostModel()
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.05] * 50 + [0.95] * 50)
    thr, cost = cm.best_threshold(y, p)
    assert 0.05 < thr <= 0.95
    assert cost == pytest.approx(50 * (cm.offer_cost + 0.5 * cm.customer_value))


def test_best_threshold_tracks_analytic_for_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 60000)
    y = rng.uniform(0, 1, 60000) < p        # perfectly calibrated
    cm = CostModel(300, 30, 0.5)
    thr, _ = cm.best_threshold(y, p)
    assert abs(thr - cm.analytic_threshold()) < 0.03


@pytest.mark.parametrize("kw", [dict(customer_value=0), dict(save_rate=0), dict(save_rate=1.5),
                                dict(offer_cost=-1)])
def test_invalid_cost_models_rejected(kw):
    with pytest.raises(ValueError):
        CostModel(**kw)
