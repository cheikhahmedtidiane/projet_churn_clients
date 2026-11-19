"""Découpage des données, features métier et pipeline de prétraitement."""

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import RANDOM_STATE, TARGET, TEST_SIZE


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
