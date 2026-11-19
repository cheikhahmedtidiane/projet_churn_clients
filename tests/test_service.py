import numpy as np
import pandas as pd
import pytest

from churn.config import BusinessParams, FEATURE_COLS, INTERNET_OPTIONS
from churn.data import load_raw
from churn.service import ChurnService, lever_is_applicable, make_client, with_changes

BASE = dict(gender="Female", SeniorCitizen="No", Partner="No", Dependents="No", tenure=2,
            PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic",
            OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
            StreamingTV="No", StreamingMovies="No", Contract="Month-to-month",
            PaperlessBilling="Yes", PaymentMethod="Electronic check", MonthlyCharges=80.0)


@pytest.fixture(scope="module")
def service():
    return ChurnService.load()


def test_make_client_enforces_consistency():
    c = make_client(**{**BASE, "InternetService": "No", "PhoneService": "No"})
    assert all(c.loc[0, col] == "No internet service" for col in INTERNET_OPTIONS)
    assert c.loc[0, "MultipleLines"] == "No phone service"
    assert c.loc[0, "TotalCharges"] == pytest.approx(160.0)
    assert set(FEATURE_COLS) <= set(c.columns)


def test_make_client_missing_field():
    fields = dict(BASE)
    del fields["Contract"]
    with pytest.raises(ValueError, match="Contract"):
        make_client(**fields)


def test_score_matches_model_and_rule(service):
    X = service.X_test.head(50)
    s = service.score(X)
    assert np.allclose(s["risque"], service.model.predict_proba(X)[:, 1])
    assert (s["contacter"] == (s["gain_espere"] > 0)).all()
    # Des hypothèses plus coûteuses ne peuvent que réduire le nombre de clients contactés
    assert service.score(X, BusinessParams(action_cost=100))["contacter"].sum() <= s["contacter"].sum()


def test_prepare_raw_file(service):
    raw = load_raw().head(5).drop(columns="Churn")
    X = service.prepare(raw)
    assert X.index.name == "customerID" and len(service.score(X)) == 5


def test_prepare_rejects_missing_columns(service):
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        service.prepare(load_raw().head(3).drop(columns=["Contract", "Churn"]))


def test_with_changes_keeps_types(service):
    x = service.prepare(make_client(**BASE))
    y = with_changes(x, {"Contract": "Two year"})
    assert y["Contract"].dtype.name == "category" and y.iloc[0]["Contract"] == "Two year"
    assert x.iloc[0]["Contract"] == "Month-to-month"  # l'original n'est pas modifié


def test_two_year_contract_lowers_risk(service):
    x = service.prepare(make_client(**BASE))
    sim = service.simulate(x, {"Contract": "Two year"})
    assert sim["risque_apres"] < sim["risque_avant"]


def test_levers_only_applicable(service):
    x = service.prepare(make_client(**{**BASE, "InternetService": "No"}))
    levers = service.recommend_levers(x)
    assert not levers["variable"].isin(INTERNET_OPTIONS).any()
    assert levers["risque_apres"].is_monotonic_increasing
    row = x.iloc[0]
    assert not lever_is_applicable(row, "Contract", "Month-to-month")


def test_top_factors_text(service):
    X = service.X_test.head(10)
    factors = service.top_factors(X)
    assert len(factors) == 10 and factors.index.equals(X.index)
