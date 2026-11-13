"""Chargement et nettoyage des données.

Implémente les décisions de l'audit de qualité (notebooks/02_qualite_donnees.ipynb).
Le même nettoyage est utilisé à l'entraînement ET dans l'application, pour éviter
tout écart entre les données vues par le modèle et celles qu'on lui soumet.
"""

from pathlib import Path

import pandas as pd

from churn.config import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    ID_COL,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    TARGET,
)

_YES_NO = {"Yes": 1, "No": 0}


def load_raw(path: Path | str = RAW_DATA_FILE) -> pd.DataFrame:
    """Lit le CSV brut, sans aucune transformation."""
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, require_target: bool = False) -> None:
    """Vérifie que toutes les colonnes nécessaires sont présentes.

    Raises:
        ValueError: avec la liste des colonnes manquantes.
    """
    required = FEATURE_COLS + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {', '.join(missing)}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie un DataFrame au format brut du dataset Telco.

    Fonctionne avec ou sans la colonne cible (données d'entraînement ou nouveaux clients).

    Étapes (cf. synthèse de l'audit) :
        1. TotalCharges converti en numérique ; vide pour un nouveau client → 0
           (s'il manque pour un client ancien → estimé par tenure × MonthlyCharges).
        2. SeniorCitizen harmonisé en No/Yes.
        3. Variables catégorielles typées en `category`.
        4. Cible encodée 1 = part, 0 = reste (si présente).
        5. customerID passé en index (si présent).
    """
    validate_columns(df)
    df = df.copy()

    # Espaces parasites dans les colonnes texte
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()

    # 1. TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_total = df["TotalCharges"].isna()
    new_client = df["tenure"].eq(0)
    df.loc[missing_total & new_client, "TotalCharges"] = 0.0
    df.loc[missing_total & ~new_client, "TotalCharges"] = (
        df["tenure"] * df["MonthlyCharges"]
    )[missing_total & ~new_client]
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # 2. SeniorCitizen 0/1 → No/Yes (accepte aussi des valeurs déjà en No/Yes)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("string").replace({"0": "No", "1": "Yes"})

    # 3. Typage catégoriel
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    # 4. Cible
    if TARGET in df.columns:
        unknown = set(df[TARGET].dropna().unique()) - set(_YES_NO)
        if unknown or df[TARGET].isna().any():
            raise ValueError(f"Valeurs de cible inattendues : {unknown or 'valeurs manquantes'}")
        df[TARGET] = df[TARGET].map(_YES_NO).astype("int8")

    # 5. Identifiant en index
    if ID_COL in df.columns:
        df = df.set_index(ID_COL)

    ordered = FEATURE_COLS + ([TARGET] if TARGET in df.columns else [])
    return df[ordered]


def load_clean(path: Path | str = RAW_DATA_FILE) -> pd.DataFrame:
    """Charge le CSV brut et retourne les données nettoyées."""
    return clean(load_raw(path))


def save_processed(df: pd.DataFrame, path: Path | str = PROCESSED_DATA_FILE) -> Path:
    """Sauvegarde les données nettoyées au format Parquet (types conservés)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path
