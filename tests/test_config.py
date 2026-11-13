"""Tests de la configuration centrale."""

import pytest

from churn.config import DEFAULT_BUSINESS, RAW_DATA_FILE, BusinessParams


def test_raw_data_file_exists():
    assert RAW_DATA_FILE.exists(), f"Dataset introuvable : {RAW_DATA_FILE}"


def test_gain_true_positive_matches_framing():
    # Partant moyen : 74,4 $/mois → 0,30 × 12 × 74,4 − 50 ≈ 217,8 $
    assert DEFAULT_BUSINESS.gain_true_positive(74.4) == pytest.approx(217.84)


def test_gain_false_positive_is_action_cost():
    assert DEFAULT_BUSINESS.gain_false_positive() == -50.0


def test_business_params_are_customizable():
    params = BusinessParams(action_cost=100.0, success_rate=0.5)
    assert params.gain_true_positive(100.0) == pytest.approx(0.5 * 12 * 100 - 100)
