"""Tests de l'évaluation, de la comparaison statistique et de la sauvegarde des modèles."""

import numpy as np
import pytest
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from churn.data import load_clean
from churn.features import build_pipeline, split_data
from churn.models import corrected_paired_ttest, evaluate_cv, load_model, make_cv, save_model


@pytest.fixture(scope="module")
def train():
    X_train, _, y_train, _ = split_data(load_clean())
    return X_train, y_train


def test_evaluate_cv_dummy_matches_prevalence(train):
    X, y = train
    res = evaluate_cv(build_pipeline(DummyClassifier(strategy="prior")), X, y,
                      cv=make_cv(n_repeats=1), n_jobs=1)
    assert res["pr_auc"] == pytest.approx(y.mean(), abs=0.01)  # modèle sans information
    assert res["roc_auc"] == pytest.approx(0.5)
    assert res["brier"] > 0  # remis dans le bon sens (positif)
    assert len(res["pr_auc_folds"]) == 5


def test_logistic_beats_dummy(train):
    X, y = train
    res = evaluate_cv(build_pipeline(LogisticRegression(max_iter=1000)), X, y,
                      cv=make_cv(n_repeats=1), n_jobs=1)
    assert res["pr_auc"] > 0.6 and res["roc_auc"] > 0.8


def test_corrected_ttest_is_more_conservative_than_naive():
    rng = np.random.default_rng(0)
    a = 0.70 + rng.normal(0, 0.02, 15)
    b = a - 0.005 + rng.normal(0, 0.005, 15)
    corrected = corrected_paired_ttest(a, b)
    naive_p = stats.ttest_rel(a, b).pvalue
    assert corrected["mean_diff"] > 0
    assert corrected["p_value"] > naive_p  # la correction réduit la significativité


def test_corrected_ttest_identical_scores():
    a = np.full(15, 0.7)
    assert corrected_paired_ttest(a, a)["p_value"] == 1.0


def test_save_and_load_roundtrip(train, tmp_path):
    X, y = train
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    save_model(pipe, "test_model", {"pr_auc_cv": 0.67}, models_dir=tmp_path)
    loaded, meta = load_model("test_model", models_dir=tmp_path)
    np.testing.assert_allclose(loaded.predict_proba(X.head(20)), pipe.predict_proba(X.head(20)))
    assert meta["pr_auc_cv"] == 0.67 and "created_at" in meta and "sklearn_version" in meta
