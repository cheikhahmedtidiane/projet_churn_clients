"""Configuration centrale du projet : chemins, constantes et hypothèses business.

Toute valeur utilisée à plusieurs endroits (notebooks, scripts, application)
est définie ici, et uniquement ici.
"""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins (construits à partir de l'emplacement de ce fichier → portables)
# src/churn/config.py → parents[2] = racine du projet
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_FILE = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed" / "churn_clean.parquet"

MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# ---------------------------------------------------------------------------
# Constantes de données et de modélisation
# ---------------------------------------------------------------------------
TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    # Profil client
    "gender", "SeniorCitizen", "Partner", "Dependents",
    # Services souscrits
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    # Contrat et facturation
    "Contract", "PaperlessBilling", "PaymentMethod",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

RANDOM_STATE = 42  # reproductibilité : mêmes résultats à chaque exécution
TEST_SIZE = 0.20   # 20 % des clients réservés à l'évaluation finale


# ---------------------------------------------------------------------------
# Hypothèses business (cf. docs/cadrage_metier.md, section 4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BusinessParams:
    """Hypothèses économiques d'une campagne de rétention.

    Attributes:
        clv_months: horizon (en mois) de revenu préservé quand un client est retenu.
        action_cost: coût d'une action de rétention par client contacté ($).
        success_rate: probabilité qu'un client sur le départ, une fois contacté, reste.
    """

    clv_months: int = 12
    action_cost: float = 50.0
    success_rate: float = 0.30

    def gain_true_positive(self, monthly_charges: float) -> float:
        """Gain d'un vrai positif : partant détecté et contacté."""
        return self.success_rate * self.clv_months * monthly_charges - self.action_cost

    def gain_false_positive(self) -> float:
        """Gain (négatif) d'un faux positif : client fidèle contacté pour rien."""
        return -self.action_cost


DEFAULT_BUSINESS = BusinessParams()
