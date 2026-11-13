"""Tests du chargement et du nettoyage des données."""

import pandas as pd
import pytest

from churn.config import CATEGORICAL_COLS, FEATURE_COLS, ID_COL, TARGET
from churn.data import clean, load_clean


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """Trois clients au format brut : nouveau client, client ancien sans total, client normal."""
    base = {
        "gender": "Female", "Partner": "Yes", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }
    rows = [
        {ID_COL: "A", "SeniorCitizen": 0, "tenure": 0, "MonthlyCharges": 50.0, "TotalCharges": " ", TARGET: "No"},
        {ID_COL: "B", "SeniorCitizen": 1, "tenure": 12, "MonthlyCharges": 80.0, "TotalCharges": " ", TARGET: "Yes"},
        {ID_COL: "C", "SeniorCitizen": 0, "tenure": 5, "MonthlyCharges": 20.0, "TotalCharges": "101.5", TARGET: "No"},
    ]
    return pd.DataFrame([{**base, **r} for r in rows])


# --- Règles de nettoyage, une par une ------------------------------------------

def test_total_charges_new_client_is_zero(raw_sample):
    assert clean(raw_sample).loc["A", "TotalCharges"] == 0.0


def test_total_charges_missing_for_old_client_is_estimated(raw_sample):
    assert clean(raw_sample).loc["B", "TotalCharges"] == pytest.approx(12 * 80.0)


def test_total_charges_valid_value_is_kept_and_numeric(raw_sample):
    out = clean(raw_sample)
    assert out.loc["C", "TotalCharges"] == pytest.approx(101.5)
    assert pd.api.types.is_float_dtype(out["TotalCharges"])


def test_senior_citizen_harmonized(raw_sample):
    out = clean(raw_sample)
    assert out["SeniorCitizen"].tolist() == ["No", "Yes", "No"]


def test_categorical_columns_typed(raw_sample):
    out = clean(raw_sample)
    assert all(isinstance(out[c].dtype, pd.CategoricalDtype) for c in CATEGORICAL_COLS)


def test_target_encoded(raw_sample):
    assert clean(raw_sample)[TARGET].tolist() == [0, 1, 0]


def test_id_becomes_index(raw_sample):
    out = clean(raw_sample)
    assert out.index.name == ID_COL
    assert ID_COL not in out.columns


# --- Robustesse (usage dans l'application) ----------------------------------------

def test_works_without_target(raw_sample):
    out = clean(raw_sample.drop(columns=TARGET))
    assert TARGET not in out.columns
    assert list(out.columns) == FEATURE_COLS


def test_missing_column_raises_clear_error(raw_sample):
    with pytest.raises(ValueError, match="Contract"):
        clean(raw_sample.drop(columns="Contract"))


def test_unexpected_target_value_raises(raw_sample):
    raw_sample.loc[0, TARGET] = "Maybe"
    with pytest.raises(ValueError, match="cible"):
        clean(raw_sample)


def test_input_not_modified(raw_sample):
    before = raw_sample.copy()
    clean(raw_sample)
    pd.testing.assert_frame_equal(raw_sample, before)


# --- Dataset réel -------------------------------------------------------------------

def test_real_dataset_is_clean():
    df = load_clean()
    assert df.shape == (7043, len(FEATURE_COLS) + 1)
    assert df.isna().sum().sum() == 0
    assert df.index.is_unique
    assert df[TARGET].mean() == pytest.approx(0.2654, abs=1e-3)


def test_parquet_round_trip_keeps_dtypes(raw_sample, tmp_path):
    from churn.data import save_processed

    out = clean(raw_sample)
    reloaded = pd.read_parquet(save_processed(out, tmp_path / "sample.parquet"))
    pd.testing.assert_frame_equal(reloaded, out)
