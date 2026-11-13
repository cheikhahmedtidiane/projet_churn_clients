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


def fmt_pct(x: float, decimals: int = 1) -> str:
    """Formate une proportion en pourcentage à la française : 0.265 → '26,5 %'."""
    return f"{x * 100:.{decimals}f}".replace(".", ",") + " %"


def fmt_int(n: float) -> str:
    """Formate un entier avec espace comme séparateur de milliers : 7043 → '7 043'."""
    return f"{int(round(n)):,}".replace(",", " ")


def plot_churn_rate(ax, table, title: str, global_rate: float) -> None:
    """Barres horizontales du taux de churn par modalité (sortie de churn_rate_table).

    La ligne pointillée matérialise le taux de churn moyen.
    """
    table = table.sort_values("churn_rate")
    labels = [str(i) for i in table.index]
    bars = ax.barh(labels, table["churn_rate"], color=ORANGE, height=0.6)
    for bar, rate, n in zip(bars, table["churn_rate"], table["clients"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{fmt_pct(rate)}  (n={fmt_int(n)})",
                va="center", fontsize=9, color=TEXT, zorder=4,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.axvline(global_rate, color=TEXT_SECONDARY, lw=1, ls="--", zorder=3)
    ax.text(global_rate, 1.0, " moyenne", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, max(table["churn_rate"].max() * 1.55, global_rate * 1.3))
    ax.xaxis.set_major_formatter(lambda v, _: fmt_pct(v, 0))
    ax.grid(axis="y", visible=False)
    ax.set_title(title, fontsize=11)
