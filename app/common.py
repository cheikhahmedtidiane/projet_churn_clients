"""Éléments partagés par les pages : service mis en cache, hypothèses business, formats."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from churn.config import DEFAULT_BUSINESS, BusinessParams  # noqa: E402
from churn.service import CATEGORY_OPTIONS, ChurnService, label, make_client  # noqa: E402
from churn.viz import set_style  # noqa: E402

set_style()

FIELD_LABELS = {
    "gender": "Genre", "SeniorCitizen": "Senior (65 ans et plus)", "Partner": "En couple",
    "Dependents": "Personnes à charge", "tenure": "Ancienneté (mois)", "PhoneService": "Ligne téléphonique",
    "MultipleLines": "Lignes multiples", "InternetService": "Internet", "OnlineSecurity": "Sécurité en ligne",
    "OnlineBackup": "Sauvegarde en ligne", "DeviceProtection": "Protection de l'appareil",
    "TechSupport": "Support technique", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming films",
    "Contract": "Contrat", "PaperlessBilling": "Facture dématérialisée", "PaymentMethod": "Moyen de paiement",
    "MonthlyCharges": "Facture mensuelle ($)",
}
GENDER_LABELS = {"Female": "Femme", "Male": "Homme"}

# Valeurs proposées par défaut dans le formulaire (profil typique d'un nouveau client)
DEFAULT_CLIENT = {"PhoneService": "Yes", "InternetService": "Fiber optic", "PaperlessBilling": "Yes",
                  "tenure": 2, "MonthlyCharges": 80.0}


@st.cache_resource(show_spinner="Chargement du modèle…")
def get_service() -> ChurnService:
    return ChurnService.load()


def business_params() -> BusinessParams:
    """Hypothèses économiques réglables dans la barre latérale (partagées entre les pages)."""
    with st.sidebar:
        st.subheader("Hypothèses de campagne")
        cost = st.slider("Coût d'une action ($)", 10, 150, int(DEFAULT_BUSINESS.action_cost), 5, key="cost",
                         help="Coût par client contacté (remise, temps de conseiller…).")
        rate = st.slider("Taux de succès (%)", 5, 60, int(round(DEFAULT_BUSINESS.success_rate * 100)), 5,
                         key="rate", help="Part des clients sur le départ qui restent après l'action.") / 100
        months = st.slider("Horizon de valeur (mois)", 6, 36, DEFAULT_BUSINESS.clv_months, 6, key="months",
                           help="Nombre de mois de facture préservés quand un client est retenu.")
        st.caption("Valeurs par défaut : hypothèses du cadrage (50 $, 30 %, 12 mois).")
    return BusinessParams(clv_months=months, action_cost=float(cost), success_rate=rate)


def option_label(col: str, value: str) -> str:
    return GENDER_LABELS.get(value, label(col, value)) if col == "gender" else label(col, value)


def client_form(defaults: dict | None = None, key: str = "form") -> pd.DataFrame | None:
    """Formulaire de saisie d'un client ; retourne le client au format brut une fois validé."""
    d = {**DEFAULT_CLIENT, **(defaults or {})}

    def select(col, container):
        # Les modalités « sans internet / sans ligne » sont déduites automatiquement (make_client)
        options = [o for o in CATEGORY_OPTIONS[col] if o not in ("No internet service", "No phone service")]
        current = d.get(col, options[0])
        current = "No" if current in ("No internet service", "No phone service") else current
        return container.selectbox(FIELD_LABELS[col], options, index=options.index(current),
                                   format_func=lambda v, c=col: option_label(c, v), key=f"{key}_{col}")

    with st.form(key):
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Profil**")
        values = {col: select(col, c1) for col in ["gender", "SeniorCitizen", "Partner", "Dependents"]}
        values["tenure"] = c1.number_input(FIELD_LABELS["tenure"], 0, 72, int(d.get("tenure", 2)), key=f"{key}_tenure")
        c2.markdown("**Services**")
        for col in ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
            values[col] = select(col, c2)
        c3.markdown("**Contrat et facturation**")
        for col in ["Contract", "PaperlessBilling", "PaymentMethod"]:
            values[col] = select(col, c3)
        values["MonthlyCharges"] = c3.number_input(FIELD_LABELS["MonthlyCharges"], 18.0, 120.0,
                                                   float(d.get("MonthlyCharges", 80.0)), 0.5, key=f"{key}_mc")
        c3.caption("Sans internet ou sans ligne téléphonique, les options correspondantes sont ignorées.")
        submitted = st.form_submit_button("Évaluer ce client", type="primary")
    return make_client(**values) if submitted else None


def fmt_pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}".replace(".", ",") + " %"


def fmt_usd(x: float, signed: bool = False) -> str:
    s = f"{abs(x):,.0f}".replace(",", " ") + " $"
    if signed:
        return ("+" if x >= 0 else "−") + s
    return ("−" if x < 0 else "") + s
