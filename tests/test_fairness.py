import numpy as np
import pytest

from churn.fairness import group_report


def test_group_report_basic_metrics():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.6, 0.3, 0.4, 0.2])
    contacted = p > 0.5
    groups = np.array(["A"] * 4 + ["B"] * 4)
    r = group_report(y, p, contacted, groups, n_boot=50)
    assert list(r.index) == ["A", "B"] and r.loc["A", "n"] == 4
    assert r.loc["A", "churn_rate"] == pytest.approx(0.5)
    assert r.loc["A", "calibration_gap"] == pytest.approx(p[:4].mean() - 0.5)
    assert r.loc["A", "recall"] == pytest.approx(1.0)
    assert r.loc["B", "false_positive_rate"] == pytest.approx(0.0)
    assert r.loc["A", "contact_rate"] == pytest.approx(0.5)


def test_calibration_interval_contains_gap():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    r = group_report(y, p, p > 0.5, np.where(rng.random(500) < 0.5, "x", "y"), n_boot=200)
    assert ((r["gap_ci_low"] <= r["calibration_gap"]) & (r["calibration_gap"] <= r["gap_ci_high"])).all()
    assert (r["gap_ci_low"] < 0).all() and (r["gap_ci_high"] > 0).all()  # modèle calibré par construction


def test_single_class_group_gives_nan_auc():
    r = group_report([0, 0, 1, 0], [0.1, 0.2, 0.7, 0.3], [False, False, True, False], ["a", "a", "b", "b"], n_boot=20)
    assert np.isnan(r.loc["a", "roc_auc"]) and np.isnan(r.loc["a", "recall"])
