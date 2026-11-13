"""Style graphique commun à tout le projet (notebooks et application).

Convention de couleur, identique partout :
    bleu   = client qui reste  (Churn = 0)
    orange = client qui part   (Churn = 1)
Palette validée pour la lisibilité par les personnes daltoniennes.
"""

import matplotlib.pyplot as plt

from churn.config import FIGURES_DIR

# Couleurs de séries
BLUE = "#2a78d6"    # reste / série unique
ORANGE = "#eb6834"  # part

CHURN_COLORS = {0: BLUE, 1: ORANGE}
CHURN_LABELS = {0: "Reste", 1: "Part"}

# Encre et surfaces (le texte n'utilise jamais la couleur d'une série)
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def set_style() -> None:
    """Applique le style du projet à matplotlib (grille discrète, axes sobres)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT,
            "legend.frameon": False,
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def save_fig(fig, name: str) -> None:
    """Sauvegarde une figure dans reports/figures/ (réutilisée dans le README)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name)
