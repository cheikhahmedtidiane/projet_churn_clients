"""Tests du découpage, des features métier et du pipeline."""

import numpy as np
import pandas as pd
import pytest

from churn.config import TARGET
from churn.data import load_clean
from sklearn.linear_model import LogisticRegression

from churn.features import ChurnFeatureEngineer, build_pipeline, split_data


@pytest.fixture(scope="module")
def clean_df():
    return load_clean()


# --- Découpage train/test --------------------------------------------------------

def test_split_sizes_and_no_overlap(clean_df):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    assert len(X_test) == pytest.approx(0.2 * len(clean_df), abs=1)
    assert len(X_train) + len(X_test) == len(clean_df)
    assert X_train.index.intersection(X_test.index).empty
    assert TARGET not in X_train.columns


def test_split_is_stratified(clean_df):
    _, _, y_train, y_test = split_data(clean_df)
    assert y_train.mean() == pytest.approx(clean_df[TARGET].mean(), abs=0.002)
    assert y_test.mean() == pytest.approx(clean_df[TARGET].mean(), abs=0.002)


def test_split_is_reproducible(clean_df):
    a = split_data(clean_df)[1].index
    b = split_data(clean_df)[1].index
    assert a.equals(b)


# --- Features métier ---------------------------------------------------------------

def test_feature_engineer_values():
    X = load_clean().drop(columns=TARGET).head(200)
    out = ChurnFeatureEngineer().fit_transform(X)
    row, src = out.iloc[0], X.iloc[0]
    expected_prot = sum(src[c] == "Yes" for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"])
    assert row["nb_protections"] == expected_prot
    assert row["log_tenure"] == pytest.approx(np.log1p(src["tenure"]))
    assert set(out["is_new_client"].unique()) <= {0, 1}
    fiber_mtm = (X["InternetService"] == "Fiber optic") & (X["Contract"] == "Month-to-month")
    assert (out["fiber_month_to_month"] == fiber_mtm.astype(int)).all()


def test_feature_engineer_drops_columns_and_names_match():
    X = load_clean().drop(columns=TARGET).head(50)
    fe = ChurnFeatureEngineer().fit(X)
    out = fe.transform(X)
    assert not {"gender", "PhoneService", "TotalCharges"} & set(out.columns)
    assert list(out.columns) == list(fe.get_feature_names_out())


def test_feature_engineer_can_be_disabled():
    X = load_clean().drop(columns=TARGET).head(50)
    out = ChurnFeatureEngineer(add_features=False, drop_cols=(), merge_no_internet=False).fit_transform(X)
    pd.testing.assert_frame_equal(out, X)


def test_no_internet_service_merged_without_information_loss():
    X = load_clean().drop(columns=TARGET)
    out = ChurnFeatureEngineer().fit_transform(X)
    assert not (out["OnlineSecurity"] == "No internet service").any()
    # L'absence d'internet reste identifiable via InternetService
    no_internet = X["OnlineSecurity"] == "No internet service"
    assert (out.loc[no_internet, "InternetService"] == "No").all()


def test_feature_engineer_does_not_modify_input():
    X = load_clean().drop(columns=TARGET).head(50)
    before = X.copy()
    ChurnFeatureEngineer().fit_transform(X)
    pd.testing.assert_frame_equal(X, before)


# --- Pipeline complet ----------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted(clean_df):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    return pipe, X_train, X_test


def test_pipeline_predicts_probabilities(fitted):
    pipe, _, X_test = fitted
    proba = pipe.predict_proba(X_test)[:, 1]
    assert proba.shape == (len(X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_preprocessing_output_is_numeric_with_readable_names(fitted):
    pipe, _, X_test = fitted
    Xt = pipe[:-1].transform(X_test)
    assert Xt.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(Xt[c]) for c in Xt.columns)
    assert "Contract_Two year" in Xt.columns and "nb_protections" in Xt.columns
    assert not any(c.startswith("gender") for c in Xt.columns)
    assert not any("No internet service" in c for c in Xt.columns)


def test_no_perfectly_duplicated_columns(fitted):
    """Aucune colonne identique à une autre (multicolinéarité parfaite)."""
    pipe, X_train, _ = fitted
    Xt = pipe[:-1].transform(X_train)
    assert not Xt.T.duplicated().any()


def test_scaler_fitted_on_train_only(fitted):
    """Pas de fuite : la moyenne du scaler est celle du jeu d'entraînement."""
    pipe, X_train, _ = fitted
    scaler = pipe.named_steps["preprocess"].named_transformers_["num"]
    idx = list(scaler.feature_names_in_).index("MonthlyCharges")
    assert scaler.mean_[idx] == pytest.approx(X_train["MonthlyCharges"].mean())


def test_unknown_category_does_not_crash(fitted):
    pipe, _, X_test = fitted
    X_new = X_test.head(3).copy()
    X_new["PaymentMethod"] = X_new["PaymentMethod"].cat.add_categories("Crypto")
    X_new.iloc[0, X_new.columns.get_loc("PaymentMethod")] = "Crypto"
    with pytest.warns(UserWarning, match="unknown categories"):
        assert pipe.predict_proba(X_new).shape == (3, 2)
