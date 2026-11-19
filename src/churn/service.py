"""Couche de service de l'application : toute la logique, aucune interface.

L'application Streamlit n'appelle que ce module. Il réutilise exactement le nettoyage,
le pipeline, la règle de décision et les explications des notebooks, ce qui garantit
que l'application calcule la même chose que l'analyse.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from churn.business import breakeven_proba, expected_gain
from churn.config import (CATEGORICAL_COLS, DEFAULT_BUSINESS, FEATURE_COLS, ID_COL,
                          INTERNET_OPTIONS, BusinessParams)
from churn.data import clean, load_clean
from churn.explain import ACTIONABLE, FEATURE_LABELS, VALUE_LABELS, ChurnExplainer, format_value
from churn.features import split_data
from churn.models import load_model

# Modalités possibles de chaque variable catégorielle (format brut du dataset)
CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{col: ["No", "Yes", "No internet service"] for col in INTERNET_OPTIONS},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"],
}

# Leviers simulés : (variable, nouvelle valeur, libellé de l'action)
LEVERS = [
    ("Contract", "One year", "Proposer un contrat 1 an"),
    ("Contract", "Two year", "Proposer un contrat 2 ans"),
    ("OnlineSecurity", "Yes", "Offrir la sécurité en ligne"),
    ("TechSupport", "Yes", "Offrir le support technique"),
    ("OnlineBackup", "Yes", "Offrir la sauvegarde en ligne"),
    ("DeviceProtection", "Yes", "Offrir la protection de l'appareil"),
    ("PaymentMethod", "Credit card (automatic)", "Passer au paiement automatique"),
]
_CONTRACT_RANK = {"Month-to-month": 0, "One year": 1, "Two year": 2}


def make_client(**fields) -> pd.DataFrame:
    """Construit un client au format brut à partir d'un formulaire, en rendant les champs cohérents.

    Sans internet, les options internet valent « No internet service » ; sans ligne
    téléphonique, MultipleLines vaut « No phone service ». TotalCharges est estimé
    (ancienneté × facture) s'il n'est pas fourni.
    """
    client = dict(fields)
    client.setdefault("TotalCharges", client["tenure"] * client["MonthlyCharges"])
    if client["InternetService"] == "No":
        for col in INTERNET_OPTIONS:
            client[col] = "No internet service"
    else:
        for col in INTERNET_OPTIONS:
            if client.get(col) == "No internet service":
                client[col] = "No"
    if client["PhoneService"] == "No":
        client["MultipleLines"] = "No phone service"
    elif client.get("MultipleLines") == "No phone service":
        client["MultipleLines"] = "No"
    missing = [c for c in FEATURE_COLS if c not in client]
    if missing:
        raise ValueError(f"Champs manquants : {', '.join(missing)}")
    return pd.DataFrame([client])


def with_changes(x: pd.DataFrame, changes: dict) -> pd.DataFrame:
    """Copie d'un client nettoyé avec certaines variables modifiées (types préservés)."""
    x = x.copy()
    for col, value in changes.items():
        if col in CATEGORICAL_COLS:
            x[col] = pd.Categorical([value] * len(x))
        else:
            x[col] = value
    return x


def lever_is_applicable(row: pd.Series, col: str, value: str) -> bool:
    """Un levier n'a de sens que s'il change quelque chose et qu'il est réalisable."""
    current = str(row[col])
    if col == "Contract":
        return _CONTRACT_RANK[value] > _CONTRACT_RANK[current]
    if col in INTERNET_OPTIONS:
        return row["InternetService"] != "No" and current == "No"
    if col == "PaymentMethod":
        return current in ("Electronic check", "Mailed check")
    return current != value


@dataclass
class ChurnService:
    model: object
    card: dict
    explainer: ChurnExplainer
    X_test: pd.DataFrame
    y_test: pd.Series

    @classmethod
    def load(cls) -> "ChurnService":
        """Charge le modèle sauvegardé et prépare l'explainer (référence : jeu d'entraînement)."""
        model, card = load_model("churn_model")
        X_train, X_test, _, y_test = split_data(load_clean())
        return cls(model, card, ChurnExplainer(model, X_train), X_test, y_test)

    # --- Préparation -------------------------------------------------------------
    @staticmethod
    def prepare(raw: pd.DataFrame) -> pd.DataFrame:
        """Valide et nettoie des clients au format brut (lève ValueError si colonnes manquantes)."""
        return clean(raw)

    # --- Prédiction et décision ---------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def score(self, X: pd.DataFrame, params: BusinessParams = DEFAULT_BUSINESS) -> pd.DataFrame:
        """Risque, gain espéré et décision pour chaque client (règle : gain espéré > 0)."""
        proba = self.predict_proba(X)
        gain = expected_gain(proba, X["MonthlyCharges"], params)
        return pd.DataFrame({
            "risque": proba,
            "facture": X["MonthlyCharges"].to_numpy(dtype=float),
            "seuil_rentabilite": breakeven_proba(X["MonthlyCharges"], params),
            "gain_espere": gain,
            "contacter": gain > 0,
        }, index=X.index)

    # --- Explications --------------------------------------------------------------
    def explain(self, x: pd.DataFrame) -> pd.DataFrame:
        return self.explainer.explain_client(x)

    def top_factors(self, X: pd.DataFrame, n: int = 3) -> pd.Series:
        """Principaux facteurs de risque de chaque client, en texte (vectorisé pour un fichier)."""
        contrib = self.explainer.grouped_values(X)
        out = {}
        for idx, row in contrib.iterrows():
            top = row[row > 0.05].sort_values(ascending=False).head(n)
            out[idx] = " · ".join(f"{FEATURE_LABELS[g]} : {format_value(g, X.loc[idx])}" for g in top.index)
        return pd.Series(out, name="facteurs_de_risque")

    # --- Simulation (what-if) -------------------------------------------------------
    def simulate(self, x: pd.DataFrame, changes: dict,
                 params: BusinessParams = DEFAULT_BUSINESS) -> dict:
        """Risque et gain espéré d'un client avant et après modification de certaines variables."""
        before = self.score(x, params).iloc[0]
        after = self.score(with_changes(x, changes), params).iloc[0]
        return {"risque_avant": before["risque"], "risque_apres": after["risque"],
                "gain_avant": before["gain_espere"], "gain_apres": after["gain_espere"]}

    def recommend_levers(self, x: pd.DataFrame, params: BusinessParams = DEFAULT_BUSINESS) -> pd.DataFrame:
        """Effet de chaque levier applicable sur le risque du client, du plus efficace au moins efficace."""
        row = x.iloc[0]
        rows = []
        for col, value, action in LEVERS:
            if not lever_is_applicable(row, col, value):
                continue
            sim = self.simulate(x, {col: value}, params)
            rows.append({"action": action, "variable": col, "nouvelle_valeur": value,
                         "risque_avant": sim["risque_avant"], "risque_apres": sim["risque_apres"],
                         "baisse_relative": 1 - sim["risque_apres"] / sim["risque_avant"]})
        table = pd.DataFrame(rows, columns=["action", "variable", "nouvelle_valeur", "risque_avant",
                                            "risque_apres", "baisse_relative"])
        return table.sort_values("risque_apres").reset_index(drop=True)


def label(col: str, value) -> str:
    """Libellé français d'une modalité brute (pour les formulaires)."""
    return VALUE_LABELS.get(str(value), str(value))


__all__ = ["ACTIONABLE", "CATEGORY_OPTIONS", "ChurnService", "ID_COL", "LEVERS", "label",
           "lever_is_applicable", "make_client", "with_changes"]
