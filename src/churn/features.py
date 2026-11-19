"""Découpage des données, features métier et pipeline de prétraitement."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import (
    DROPPED_COLS,
    INTERNET_OPTIONS,
    NEW_CLIENT_MAX_TENURE,
    PROTECTION_SERVICES,
    RANDOM_STATE,
    SELECTED_FEATURES,
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


# Calcul de chaque feature candidate (à partir des données nettoyées)
FEATURE_BUILDERS = {
    "log_tenure": lambda X: np.log1p(X["tenure"]),
    "nb_protections": lambda X: (X[PROTECTION_SERVICES] == "Yes").sum(axis=1),
    "nb_streaming": lambda X: (X[STREAMING_SERVICES] == "Yes").sum(axis=1),
    "is_new_client": lambda X: (X["tenure"] <= NEW_CLIENT_MAX_TENURE).astype(int),
    "fiber_month_to_month": lambda X: (
        (X["InternetService"] == "Fiber optic") & (X["Contract"] == "Month-to-month")
    ).astype(int),
}


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Ajoute les features métier issues de l'EDA et retire les variables inutiles.

    Transformateur sans état (rien n'est appris au `fit`) : il ne peut pas provoquer
    de fuite de données, et s'applique à l'identique à l'entraînement et en production.

    Args:
        features: features métier à ajouter (parmi FEATURE_BUILDERS). Par défaut, la
            sélection validée par l'étude d'ablation ; () pour n'en ajouter aucune.
        drop_cols: variables à retirer.
        merge_no_internet: recode « No internet service » en « No » pour les services
            optionnels. Sans cela, l'absence d'internet est encodée 7 fois à l'identique
            (multicolinéarité parfaite) ; elle reste portée par InternetService = No.
    """

    def __init__(self, features: tuple = tuple(SELECTED_FEATURES),
                 drop_cols: tuple = tuple(DROPPED_COLS), merge_no_internet: bool = True):
        self.features = features
        self.drop_cols = drop_cols
        self.merge_no_internet = merge_no_internet

    def fit(self, X: pd.DataFrame, y=None):
        unknown = set(self.features) - set(FEATURE_BUILDERS)
        if unknown:
            raise ValueError(f"Features inconnues : {sorted(unknown)}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.merge_no_internet:
            for col in INTERNET_OPTIONS:
                X[col] = X[col].astype(str).replace("No internet service", "No").astype("category")
        for name in self.features:
            X[name] = FEATURE_BUILDERS[name](X)
        return X.drop(columns=[c for c in self.drop_cols if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        names = [c for c in self.feature_names_in_ if c not in self.drop_cols]
        names += [c for c in self.features if c not in self.drop_cols]
        return np.asarray(names, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Encodage et mise à l'échelle, appliqués après le feature engineering.

    - Variables numériques (dont les features créées) : StandardScaler.
    - Variables catégorielles : One-Hot Encoding. Une seule colonne pour les variables
      binaires ; une modalité inconnue (jamais vue à l'entraînement) est ignorée
      au lieu de faire planter la prédiction.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), make_column_selector(dtype_include="number")),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
             make_column_selector(dtype_include="category")),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(model, features: tuple = tuple(SELECTED_FEATURES),
                   drop_cols: tuple = tuple(DROPPED_COLS),
                   merge_no_internet: bool = True) -> Pipeline:
    """Pipeline complet : features métier → prétraitement → modèle.

    Toutes les étapes sont ajustées uniquement sur les données d'entraînement
    (y compris à chaque pli de validation croisée) : aucune fuite de données.
    Le pipeline entraîné prend en entrée des données nettoyées par `churn.data.clean`.
    """
    pipeline = Pipeline([
        ("features", ChurnFeatureEngineer(features=features, drop_cols=drop_cols,
                                          merge_no_internet=merge_no_internet)),
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])
    return pipeline.set_output(transform="pandas")
