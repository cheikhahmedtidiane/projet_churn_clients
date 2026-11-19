"""Découpage des données, features métier et pipeline de prétraitement."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from churn.config import (
    DROPPED_COLS,
    ENGINEERED_FLAGS,
    ENGINEERED_NUMERIC,
    NEW_CLIENT_MAX_TENURE,
    PROTECTION_SERVICES,
    RANDOM_STATE,
    STREAMING_SERVICES,
    TARGET,
    TEST_SIZE,
)


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE,
               random_state: int = RANDOM_STATE):
    """Découpe les données en jeux d'entraînement et de test, stratifiés sur la cible.

    Le jeu de test est mis de côté jusqu'à l'évaluation finale : aucune décision
    (features, hyperparamètres, seuil) ne doit être prise en le regardant.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Ajoute les features métier issues de l'EDA et retire les variables inutiles.

    Transformateur sans état (rien n'est appris au `fit`) : il ne peut pas provoquer
    de fuite de données, et s'applique à l'identique à l'entraînement et en production.

    Args:
        add_features: ajoute les features métier (désactivable pour comparer).
        drop_cols: variables à retirer.
    """

    def __init__(self, add_features: bool = True, drop_cols: tuple = tuple(DROPPED_COLS)):
        self.add_features = add_features
        self.drop_cols = drop_cols

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.add_features:
            X["log_tenure"] = np.log1p(X["tenure"])
            X["nb_protections"] = (X[PROTECTION_SERVICES] == "Yes").sum(axis=1)
            X["nb_streaming"] = (X[STREAMING_SERVICES] == "Yes").sum(axis=1)
            X["is_new_client"] = (X["tenure"] <= NEW_CLIENT_MAX_TENURE).astype(int)
            X["fiber_month_to_month"] = (
                (X["InternetService"] == "Fiber optic") & (X["Contract"] == "Month-to-month")
            ).astype(int)
        return X.drop(columns=[c for c in self.drop_cols if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        names = [c for c in self.feature_names_in_ if c not in self.drop_cols]
        if self.add_features:
            names += ENGINEERED_NUMERIC + ENGINEERED_FLAGS
        return np.asarray(names, dtype=object)
