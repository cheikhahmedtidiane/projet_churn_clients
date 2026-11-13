"""Tests des fonctions d'analyse exploratoire."""

import numpy as np
import pandas as pd
import pytest

from churn.eda import categorical_association, churn_rate_table, cramers_v, numeric_association


@pytest.fixture
def toy() -> pd.DataFrame:
    # Segment A : 1 partant sur 4 ; segment B : 3 partants sur 4 → taux global 50 %
    return pd.DataFrame({
        "segment": ["A"] * 4 + ["B"] * 4,
        "montant": [10, 20, 30, 40, 50, 60, 70, 80],
        "Churn": [0, 0, 0, 1, 1, 1, 1, 0],
    })


def test_churn_rate_table_values(toy):
    t = churn_rate_table(toy, "segment")
    assert t.loc["B", "churn_rate"] == pytest.approx(0.75)
    assert t.loc["A", "churners"] == 1
    assert t.loc["B", "lift"] == pytest.approx(1.5)
    assert t["share"].sum() == pytest.approx(1.0)
    assert t.index[0] == "B"  # trié du plus risqué au moins risqué


def test_cramers_v_perfect_and_null_association():
    x = pd.Series(["a", "b"] * 50)
    assert cramers_v(x, x) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    noise = pd.Series(rng.integers(0, 2, 10_000))
    independent = pd.Series(rng.integers(0, 2, 10_000))
    assert cramers_v(noise, independent) < 0.05


def test_categorical_association_columns(toy):
    out = categorical_association(toy, ["segment"])
    assert {"chi2", "p_value", "cramers_v", "significatif"} <= set(out.columns)


def test_numeric_association_direction(toy):
    out = numeric_association(toy, ["montant"])
    # Les partants ont des montants plus élevés → corrélation positive
    assert out.loc["montant", "rank_biserial"] > 0
    assert out.loc["montant", "mediane_part"] > out.loc["montant", "mediane_reste"]
