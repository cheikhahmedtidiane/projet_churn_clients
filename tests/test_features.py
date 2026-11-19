"""Tests du découpage, des features métier et du pipeline."""

import pytest

from churn.config import TARGET
from churn.data import load_clean
from churn.features import split_data


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
