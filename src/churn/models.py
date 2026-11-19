"""Évaluation, comparaison statistique et sauvegarde des modèles."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from churn.config import MODELS_DIR, RANDOM_STATE

# Métriques de validation croisée. Brier et log loss sont « négatifs » dans scikit-learn
# (convention : plus grand = meilleur) ; on les remet dans le bon sens à l'affichage.
CV_SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "brier": "neg_brier_score",
}
_NEGATED = {"brier"}


def make_cv(n_splits: int = 5, n_repeats: int = 3, random_state: int = RANDOM_STATE):
    """Validation croisée stratifiée répétée : mêmes plis pour tous les modèles comparés."""
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=random_state)


def evaluate_cv(estimator, X, y, cv=None, n_jobs: int = -1) -> dict:
    """Évalue un estimateur en validation croisée.

    Returns:
        dict avec, pour chaque métrique : la moyenne, l'écart-type et les scores par pli
        (`<métrique>_folds`, utiles pour une comparaison appariée entre modèles).
    """
    cv = cv or make_cv()
    raw = cross_validate(estimator, X, y, cv=cv, scoring=CV_SCORING, n_jobs=n_jobs)
    result = {"fit_time": float(raw["fit_time"].mean())}
    for name in CV_SCORING:
        folds = raw[f"test_{name}"]
        folds = -folds if name in _NEGATED else folds
        result[name] = float(folds.mean())
        result[f"{name}_std"] = float(folds.std())
        result[f"{name}_folds"] = folds
    return result


def corrected_paired_ttest(scores_a, scores_b, test_train_ratio: float = 0.25) -> dict:
    """Compare deux modèles évalués sur les MÊMES plis (t-test corrigé de Nadeau & Bengio, 2003).

    Les plis de validation croisée se chevauchent : les scores ne sont pas indépendants
    et un t-test classique surestime la significativité. La correction gonfle la
    variance du facteur (1/k + n_test/n_train).

    Args:
        scores_a, scores_b: scores par pli (mêmes plis, même ordre).
        test_train_ratio: n_test / n_train d'un pli (0,25 pour 5 plis).

    Returns:
        dict : écart moyen (a - b), statistique t, p-value bilatérale, part des plis gagnés par a.
    """
    diff = np.asarray(scores_a) - np.asarray(scores_b)
    k = len(diff)
    var = diff.var(ddof=1)
    if var == 0:
        return {"mean_diff": float(diff.mean()), "t_stat": np.nan, "p_value": 1.0,
                "win_rate": float((diff > 0).mean())}
    t_stat = diff.mean() / np.sqrt((1 / k + test_train_ratio) * var)
    p_value = 2 * stats.t.sf(abs(t_stat), df=k - 1)
    return {"mean_diff": float(diff.mean()), "t_stat": float(t_stat),
            "p_value": float(p_value), "win_rate": float((diff > 0).mean())}


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    """Tableau récapitulatif (une ligne par modèle) à partir des sorties de `evaluate_cv`."""
    cols = ["pr_auc", "pr_auc_std", "roc_auc", "brier", "fit_time"]
    table = pd.DataFrame({name: {c: r[c] for c in cols} for name, r in results.items()}).T
    return table.sort_values("pr_auc", ascending=False)


def save_model(pipeline, name: str, metadata: dict, models_dir: Path | str = MODELS_DIR) -> Path:
    """Sauvegarde un pipeline entraîné et sa fiche de métadonnées (JSON).

    Fichiers créés : <name>.joblib et <name>.json.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.joblib"
    joblib.dump(pipeline, model_path)

    card = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        **metadata,
    }
    (models_dir / f"{name}.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return model_path


def load_model(name: str, models_dir: Path | str = MODELS_DIR):
    """Charge un pipeline sauvegardé et sa fiche de métadonnées."""
    models_dir = Path(models_dir)
    pipeline = joblib.load(models_dir / f"{name}.joblib")
    metadata = json.loads((models_dir / f"{name}.json").read_text(encoding="utf-8"))
    return pipeline, metadata
