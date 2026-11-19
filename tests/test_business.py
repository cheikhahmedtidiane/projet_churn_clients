import numpy as np
import pytest

from churn.business import (
    best_threshold, breakeven_proba, campaign_gain, expected_gain,
    lift_table, targeting_curve, threshold_curve,
)
from churn.config import BusinessParams

PARAMS = BusinessParams(clv_months=12, action_cost=50.0, success_rate=0.30)


def test_expected_gain_formula():
    # 0,5 × 0,3 × 12 × 100 − 50 = 130
    assert expected_gain([0.5], [100.0], PARAMS)[0] == pytest.approx(130.0)


def test_breakeven_gives_zero_expected_gain():
    mc = np.array([20.0, 70.0, 110.0])
    p = breakeven_proba(mc, PARAMS)
    assert np.allclose(expected_gain(p, mc, PARAMS), 0.0)
    assert p[0] > p[2]  # un client qui paie peu exige un risque plus élevé


def test_campaign_gain_matches_matrix():
    y = [1, 0, 1, 0]
    targeted = [True, True, False, False]
    mc = [100.0, 80.0, 60.0, 40.0]
    res = campaign_gain(y, targeted, mc, PARAMS)
    # vrai positif : 0,3 × 12 × 100 − 50 = 310 ; faux positif : −50
    assert res["net_gain"] == pytest.approx(260.0)
    assert (res["true_positives"], res["false_positives"]) == (1, 1)
    assert res["precision"] == pytest.approx(0.5)
    assert res["recall"] == pytest.approx(0.5)


def test_nobody_targeted_is_zero():
    res = campaign_gain([1, 0], [False, False], [50.0, 50.0], PARAMS)
    assert res["net_gain"] == 0 and res["n_targeted"] == 0 and np.isnan(res["precision"])


def test_threshold_curve_and_best_threshold():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.7, 0.4, 0.1])
    mc = np.full(4, 100.0)
    curve = threshold_curve(y, proba, mc, thresholds=[0.0, 0.5, 0.95], params=PARAMS)
    assert curve.loc[0.0, "n_targeted"] == 4
    assert curve.loc[0.95, "net_gain"] == 0
    assert best_threshold(curve) == 0.5  # ne contacte que les deux partants


def test_targeting_curve_perfect_ranking():
    y = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.9, 0.2, 0.8])
    curve = targeting_curve(y, score, np.full(4, 100.0), PARAMS)
    assert curve["share_churners_captured"].iloc[1] == pytest.approx(1.0)
    assert curve["lift"].iloc[0] == pytest.approx(2.0)
    assert curve["net_gain"].iloc[-1] == pytest.approx(2 * 360 - 4 * 50)


def test_lift_table_sums():
    rng = np.random.default_rng(0)
    proba = rng.random(200)
    y = (rng.random(200) < proba).astype(int)
    table = lift_table(y, proba)
    assert len(table) == 10 and table["clients"].sum() == 200
    assert table["cum_share_churners"].iloc[-1] == pytest.approx(1.0)
    assert table["churn_rate"].iloc[0] > table["churn_rate"].iloc[-1]
