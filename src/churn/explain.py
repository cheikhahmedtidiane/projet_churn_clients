"""Explicabilité du modèle (SHAP) : pourquoi un client est-il jugé à risque ?

Pour un modèle linéaire, les valeurs SHAP sont exactes : la contribution d'une colonne
vaut coefficient × (valeur du client − valeur moyenne du jeu de référence), en log-odds.
Les contributions s'additionnent exactement au score du modèle :

    log-odds(client) = valeur de base + somme des contributions

Le modèle voit 27 colonnes techniques (one-hot, log…). Pour un lecteur métier, on les
regroupe par variable d'origine (« Contrat », « Ancienneté »…) en sommant leurs
contributions, ce qui préserve l'additivité.
"""

import numpy as np
import pandas as pd
import shap

from churn.config import CATEGORICAL_COLS

# Variable d'origine (ou feature métier) → libellé lisible
FEATURE_LABELS = {
    "tenure": "Ancienneté",
    "MonthlyCharges": "Facture mensuelle",
    "fiber_month_to_month": "Fibre + contrat mensuel",
    "SeniorCitizen": "Senior",
    "Partner": "En couple",
    "Dependents": "Personnes à charge",
    "MultipleLines": "Lignes multiples",
    "InternetService": "Internet",
    "OnlineSecurity": "Sécurité en ligne",
    "OnlineBackup": "Sauvegarde en ligne",
    "DeviceProtection": "Protection de l'appareil",
    "TechSupport": "Support technique",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming films",
    "Contract": "Contrat",
    "PaperlessBilling": "Facture dématérialisée",
    "PaymentMethod": "Moyen de paiement",
}

# En dessous de ce seuil (log-odds), une contribution est jugée négligeable
NEGLIGIBLE = 0.01

# Colonnes dérivées rattachées à une variable d'origine
_DERIVED = {"log_tenure": "tenure"}

VALUE_LABELS = {
    "Yes": "Oui", "No": "Non",
    "Month-to-month": "Mensuel", "One year": "1 an", "Two year": "2 ans",
    "Fiber optic": "Fibre optique", "DSL": "DSL",
    "No phone service": "Pas de ligne téléphonique",
    "Electronic check": "Chèque électronique", "Mailed check": "Chèque postal",
    "Bank transfer (automatic)": "Virement automatique",
    "Credit card (automatic)": "Carte bancaire automatique",
}


def feature_group(column: str) -> str:
    """Variable d'origine d'une colonne transformée (ex. 'Contract_Two year' → 'Contract')."""
    if column in _DERIVED:
        return _DERIVED[column]
    for cat in CATEGORICAL_COLS:
        if column.startswith(cat + "_"):
            return cat
    return column


def format_value(group: str, x: pd.Series) -> str:
    """Valeur lisible d'une variable d'origine pour un client."""
    if group == "tenure":
        return f"{int(x['tenure'])} mois"
    if group == "MonthlyCharges":
        return f"{x['MonthlyCharges']:.2f} $".replace(".", ",")
    if group == "fiber_month_to_month":
        both = x["InternetService"] == "Fiber optic" and x["Contract"] == "Month-to-month"
        return "Oui" if both else "Non"
    value = str(x[group])
    return VALUE_LABELS.get(value, value)


class ChurnExplainer:
    """Explications SHAP d'un pipeline `features → preprocess → régression logistique`.

    Args:
        pipeline: pipeline entraîné (dernière étape : modèle linéaire).
        background: clients de référence (données brutes), en pratique le jeu d'entraînement.
            Les contributions sont mesurées par rapport au « client moyen » de ce jeu.
    """

    def __init__(self, pipeline, background: pd.DataFrame):
        self.preprocess = pipeline[:-1]
        self.model = pipeline[-1]
        reference = self.preprocess.transform(background)
        self.columns = list(reference.columns)
        self.groups = pd.Series({c: feature_group(c) for c in self.columns})
        masker = shap.maskers.Independent(reference, max_samples=len(reference))
        self._explainer = shap.LinearExplainer(self.model, masker)

    @property
    def base_value(self) -> float:
        """Log-odds du client de référence (avant toute contribution)."""
        return float(np.ravel(self._explainer.expected_value)[0])

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Contributions (log-odds) par colonne transformée, une ligne par client."""
        values = self._explainer.shap_values(self.preprocess.transform(X))
        return pd.DataFrame(values, index=X.index, columns=self.columns)

    def grouped_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Contributions regroupées par variable d'origine (somme des colonnes)."""
        return self.shap_values(X).T.groupby(self.groups).sum().T

    def explain_client(self, x: pd.DataFrame) -> pd.DataFrame:
        """Explication d'un client (DataFrame d'une ligne), triée par impact décroissant.

        Colonnes : libellé, valeur lisible, contribution (log-odds), sens de l'effet.
        """
        if len(x) != 1:
            raise ValueError("explain_client attend exactement un client.")
        contrib = self.grouped_values(x).iloc[0]
        row = x.iloc[0]
        table = pd.DataFrame({
            "variable": contrib.index,
            "libellé": [FEATURE_LABELS.get(g, g) for g in contrib.index],
            "valeur": [format_value(g, row) for g in contrib.index],
            "contribution": contrib.to_numpy(),
        })
        table["effet"] = np.select(
            [table["contribution"].abs() < NEGLIGIBLE, table["contribution"] > 0],
            ["effet négligeable", "augmente le risque"], "réduit le risque")
        return table.reindex(table["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    def reasons(self, x: pd.DataFrame, top: int = 3, min_abs: float = 0.05) -> dict:
        """Principales raisons, en français : facteurs de risque et facteurs protecteurs."""
        table = self.explain_client(x)
        table = table[table["contribution"].abs() >= min_abs]
        phrase = lambda r: f"{r['libellé']} : {r['valeur']}"
        return {
            "risque": [phrase(r) for _, r in table[table["contribution"] > 0].head(top).iterrows()],
            "protection": [phrase(r) for _, r in table[table["contribution"] < 0].head(top).iterrows()],
        }
