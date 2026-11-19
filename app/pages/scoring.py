import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from churn.config import FEATURE_COLS, ID_COL, ROOT_DIR
from churn.viz import AQUA, BLUE, SURFACE, TEXT, TEXT_SECONDARY
from common import business_params, fmt_pct, fmt_usd, get_service

SAMPLE_FILE = ROOT_DIR / "data" / "sample" / "nouveaux_clients.csv"

service = get_service()
params = business_params()

st.title("Scoring d'un fichier de clients")
st.caption("Importez une liste de clients : l'application calcule le risque de chacun, le gain espéré d'un "
           "appel, et produit la liste d'appels priorisée.")

# --- 1. Données -------------------------------------------------------------------------
with st.expander("Format attendu du fichier", icon=":material/description:"):
    st.markdown(f"Fichier **CSV** au format du dataset IBM Telco : une ligne par client, colonnes "
                f"`{ID_COL}` (recommandée) et : " + ", ".join(f"`{c}`" for c in FEATURE_COLS) + ". "
                "Une colonne `Churn` éventuelle est ignorée.")
    st.download_button("Télécharger un modèle de fichier (5 clients)",
                       pd.read_csv(SAMPLE_FILE).head(5).to_csv(index=False).encode("utf-8"),
                       "modele_clients.csv", "text/csv")

source = st.radio("Données", ["Fichier d'exemple (1 409 clients)", "Importer mon fichier"], horizontal=True)
if source == "Importer mon fichier":
    uploaded = st.file_uploader("Fichier CSV", type="csv")
    if uploaded is None:
        st.info("Importez un fichier CSV pour lancer le scoring.")
        st.stop()
    raw = pd.read_csv(uploaded)
else:
    raw = pd.read_csv(SAMPLE_FILE)
    st.caption("Clients du jeu de test, que le modèle n'a jamais vus, présentés comme de nouveaux clients.")

try:
    X = service.prepare(raw)
except ValueError as err:
    st.error(f"Fichier invalide : {err}")
    st.stop()
if X.index.name != ID_COL:
    X.index = pd.Index([f"client_{i + 1}" for i in range(len(X))], name=ID_COL)


@st.cache_data(show_spinner="Scoring des clients…")
def score_file(X: pd.DataFrame, cost: float, rate: float, months: int) -> pd.DataFrame:
    from churn.config import BusinessParams
    p = BusinessParams(clv_months=months, action_cost=cost, success_rate=rate)
    scored = service.score(X, p).join(service.top_factors(X))
    return scored.sort_values("gain_espere", ascending=False)


scored = score_file(X, params.action_cost, params.success_rate, params.clv_months)
n = len(scored)
n_profitable = int(scored["contacter"].sum())

# --- 2. Qui contacter ? -------------------------------------------------------------------
st.subheader("Qui contacter ?")
mode = st.radio("Règle de ciblage",
                ["Tous les clients rentables (gain espéré > 0)", "Budget limité (nombre d'appels)"],
                horizontal=True)
if mode.startswith("Tous"):
    k = n_profitable
else:
    k = st.slider("Nombre de clients à contacter", 1, n, min(max(1, n // 5), n), 1,
                  help="Les clients sont pris dans l'ordre décroissant du gain espéré.")
selected = scored.head(k)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clients à contacter", f"{k:,}".replace(",", " "), help=f"Sur {n:,} clients.".replace(",", " "))
c2.metric("Coût de la campagne", fmt_usd(k * params.action_cost))
c3.metric("Gain net espéré", fmt_usd(selected["gain_espere"].sum(), signed=True))
c4.metric("Départs évités (espérance)", f"{(selected['risque'] * params.success_rate).sum():.0f}",
          help="Somme des risques des clients contactés × taux de succès.")
if not mode.startswith("Tous") and k > n_profitable:
    st.warning(f"Au-delà de {n_profitable} appels, chaque client supplémentaire a un gain espéré négatif : "
               "la campagne rapporterait moins.")

# Courbe du gain espéré cumulé selon le nombre d'appels
cum = scored["gain_espere"].cumsum().to_numpy()
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(np.arange(1, n + 1), cum / 1e3, color=BLUE, lw=2)
ax.axvline(k, color=AQUA, lw=2, ls="--")
ax.plot(k, cum[k - 1] / 1e3, "o", color=AQUA, markersize=8)
ax.annotate(f"{k} appels : {fmt_usd(cum[k - 1])}", (k, cum[k - 1] / 1e3), xytext=(10, -28),
            textcoords="offset points", fontsize=9, color=TEXT,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
ax.set_xlim(0, n)
ax.set_ylim(bottom=min(0, cum.min() / 1e3))
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f} k$")
ax.set_xlabel("Nombre de clients contactés (par gain espéré décroissant)")
ax.set_ylabel("Gain net espéré", color=TEXT_SECONDARY)
st.pyplot(fig, width="stretch")
plt.close(fig)

# --- 3. Liste d'appels --------------------------------------------------------------------
st.subheader("Liste d'appels priorisée")
calls = selected.reset_index().rename(columns={
    ID_COL: "Client", "risque": "Risque", "facture": "Facture ($)", "seuil_rentabilite": "Seuil rentable",
    "gain_espere": "Gain espéré ($)", "facteurs_de_risque": "Principaux facteurs de risque"})
calls.insert(0, "Rang", range(1, len(calls) + 1))
calls = calls.drop(columns="contacter")
st.dataframe(
    calls, hide_index=True, width="stretch", height=420,
    column_config={
        "Risque": st.column_config.ProgressColumn("Risque", format="percent", min_value=0, max_value=1),
        "Seuil rentable": st.column_config.NumberColumn(format="percent"),
        "Facture ($)": st.column_config.NumberColumn(format="%.2f"),
        "Gain espéré ($)": st.column_config.NumberColumn(format="%+.0f"),
        "Principaux facteurs de risque": st.column_config.TextColumn(width="large"),
    })
export = calls.copy()
export["Risque"] = export["Risque"].round(4)
export["Seuil rentable"] = export["Seuil rentable"].round(4)
export["Gain espéré ($)"] = export["Gain espéré ($)"].round(2)
st.download_button(f"Télécharger la liste d'appels ({k} clients, CSV)",
                   export.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                   "liste_appels.csv", "text/csv", type="primary",
                   help="Séparateur « ; » et virgule décimale : s'ouvre directement dans Excel en français.")
st.caption(f"Hypothèses : action à {params.action_cost:.0f} $, taux de succès {fmt_pct(params.success_rate, 0)}, "
           f"horizon {params.clv_months} mois (réglables dans la barre latérale).")
