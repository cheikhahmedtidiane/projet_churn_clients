import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.data import load_clean
from churn.explain import FEATURE_LABELS, ChurnExplainer, feature_group
from churn.features import build_pipeline, split_data


@pytest.fixture(scope="module")
def fitted():
    X_train, X_test, y_train, _ = split_data(load_clean())
    pipe = build_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    return pipe, ChurnExplainer(pipe, X_train.head(500)), X_test.head(20)


def test_feature_group_mapping():
    assert feature_group("Contract_Two year") == "Contract"
    assert feature_group("MultipleLines_No phone service") == "MultipleLines"
    assert feature_group("log_tenure") == "tenure"
    assert feature_group("MonthlyCharges") == "MonthlyCharges"


def test_shap_values_are_additive(fitted):
    pipe, explainer, X = fitted
    total = explainer.base_value + explainer.shap_values(X).sum(axis=1)
    assert np.allclose(total, pipe.decision_function(X))


def test_grouping_preserves_additivity(fitted):
    _, explainer, X = fitted
    assert np.allclose(explainer.grouped_values(X).sum(axis=1), explainer.shap_values(X).sum(axis=1))


def test_every_group_has_a_french_label(fitted):
    _, explainer, X = fitted
    assert set(explainer.grouped_values(X).columns) <= set(FEATURE_LABELS)


def test_explain_client_sorted_and_readable(fitted):
    _, explainer, X = fitted
    table = explainer.explain_client(X.iloc[[0]])
    assert list(table.columns) == ["variable", "libellé", "valeur", "contribution", "effet"]
    assert table["contribution"].abs().is_monotonic_decreasing
    assert table.loc[table["variable"] == "tenure", "valeur"].iloc[0].endswith("mois")
    assert set(table["effet"]) <= {"augmente le risque", "réduit le risque", "effet négligeable"}
    assert (table.loc[table["contribution"] > 0.01, "effet"] == "augmente le risque").all()


def test_explain_client_requires_one_row(fitted):
    _, explainer, X = fitted
    with pytest.raises(ValueError):
        explainer.explain_client(X.head(2))


def test_reasons_split_by_direction(fitted):
    _, explainer, X = fitted
    r = explainer.reasons(X.iloc[[0]], top=2)
    assert set(r) == {"risque", "protection"}
    assert len(r["risque"]) <= 2 and len(r["protection"]) <= 2
    assert all(" : " in s for s in r["risque"] + r["protection"])
