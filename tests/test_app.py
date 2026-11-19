"""Tests de fumée de l'application : chaque page s'exécute sans erreur."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
# Lancée seule, une page n'a pas app/ dans son chemin d'import (le point d'entrée s'en charge en production)
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
PAGES = ["accueil", "fiche_client", "simulation", "scoring", "tableau_de_bord"]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_error(page):
    at = AppTest.from_file(str(APP_DIR / "pages" / f"{page}.py"), default_timeout=60).run()
    assert not at.exception, at.exception


def test_client_page_shows_decision():
    at = AppTest.from_file(str(APP_DIR / "pages" / "fiche_client.py"), default_timeout=60).run()
    labels = [m.label for m in at.metric]
    assert {"Risque de départ", "Contacter ?"} <= set(labels)


def test_dashboard_reacts_to_cost():
    at = AppTest.from_file(str(APP_DIR / "pages" / "tableau_de_bord.py"), default_timeout=60).run()
    gain_default = at.metric[0].value
    at.slider(key="cost").set_value(150).run()
    assert not at.exception and at.metric[0].value != gain_default
