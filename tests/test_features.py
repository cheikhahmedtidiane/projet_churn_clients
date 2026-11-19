"""Tests du découpage, des features métier et du pipeline."""

import numpy as np
import pandas as pd
import pytest

from churn.config import TARGET
from churn.data import load_clean
from churn.features import ChurnFeatureEngineer, split_data


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
    out = ChurnFeatureEngineer(add_features=False, drop_cols=()).fit_transform(X)
    assert list(out.columns) == list(X.columns)


def test_feature_engineer_does_not_modify_input():
    X = load_clean().drop(columns=TARGET).head(50)
    before = X.copy()
    ChurnFeatureEngineer().fit_transform(X)
    pd.testing.assert_frame_equal(X, before)
