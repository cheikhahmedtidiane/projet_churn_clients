"""Point d'entrée de l'application Streamlit.

Lancement local :  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Rend le package src/churn importable, y compris sur Streamlit Community Cloud
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Churn — Campagne de rétention", page_icon=":material/insights:",
                   layout="wide")

pages = [
    st.Page("pages/accueil.py", title="Accueil", icon=":material/home:", default=True),
    st.Page("pages/fiche_client.py", title="Fiche client", icon=":material/person_search:"),
    st.Page("pages/simulation.py", title="Simulation de leviers", icon=":material/tune:"),
    st.Page("pages/scoring.py", title="Scoring d'un fichier", icon=":material/upload_file:"),
    st.Page("pages/tableau_de_bord.py", title="Tableau de bord", icon=":material/monitoring:"),
]
st.navigation(pages).run()
