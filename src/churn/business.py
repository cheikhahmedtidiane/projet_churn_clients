"""Traduction des prédictions en décisions et en dollars (campagne de rétention).

Matrice de gain, par rapport à « ne contacter personne » :
    partant contacté (vrai positif)  : success_rate × clv_months × MonthlyCharges − action_cost
    fidèle contacté  (faux positif)  : − action_cost
    client non contacté               : 0

Le gain d'un vrai positif est une espérance : seule une fraction `success_rate`
des partants contactés reste réellement.
"""

import numpy as np
import pandas as pd

from churn.config import DEFAULT_BUSINESS, BusinessParams


def _as_arrays(*arrays):
    return [np.asarray(a, dtype=float) for a in arrays]


def expected_gain(proba, monthly_charges, params: BusinessParams = DEFAULT_BUSINESS) -> np.ndarray:
    """Gain espéré si l'on contacte chaque client : p × taux de succès × CLV − coût."""
    proba, mc = _as_arrays(proba, monthly_charges)
    return proba * params.success_rate * params.clv_months * mc - params.action_cost


def breakeven_proba(monthly_charges, params: BusinessParams = DEFAULT_BUSINESS) -> np.ndarray:
    """Risque minimal à partir duquel contacter un client est rentable en espérance."""
    (mc,) = _as_arrays(monthly_charges)
    return params.action_cost / (params.success_rate * params.clv_months * mc)


def campaign_gain(y_true, targeted, monthly_charges,
                  params: BusinessParams = DEFAULT_BUSINESS) -> dict:
    """Bilan d'une campagne : qui est contacté, combien de partants captés, gain net."""
    y, t, mc = _as_arrays(y_true, targeted, monthly_charges)
    t = t.astype(bool)
    tp = t & (y == 1)
    fp = t & (y == 0)
    gain = (params.success_rate * params.clv_months * mc[tp]).sum() - params.action_cost * t.sum()
    n_targeted = int(t.sum())
    return {
        "n_targeted": n_targeted,
        "share_targeted": n_targeted / len(y),
        "true_positives": int(tp.sum()),
        "false_positives": int(fp.sum()),
        "precision": tp.sum() / n_targeted if n_targeted else np.nan,
        "recall": tp.sum() / (y == 1).sum(),
        "cost": params.action_cost * n_targeted,
        "net_gain": float(gain),
    }


def threshold_curve(y_true, proba, monthly_charges, thresholds=None,
                    params: BusinessParams = DEFAULT_BUSINESS) -> pd.DataFrame:
    """Bilan de campagne pour chaque seuil de probabilité (contacter si p ≥ seuil)."""
    thresholds = np.round(np.arange(0.0, 1.0001, 0.01), 2) if thresholds is None else thresholds
    proba = np.asarray(proba, dtype=float)
    rows = [{"threshold": s, **campaign_gain(y_true, proba >= s, monthly_charges, params)}
            for s in thresholds]
    return pd.DataFrame(rows).set_index("threshold")


def best_threshold(curve: pd.DataFrame) -> float:
    """Seuil qui maximise le gain net (le plus bas en cas d'égalité)."""
    return float(curve["net_gain"].idxmax())


def targeting_curve(y_true, score, monthly_charges,
                    params: BusinessParams = DEFAULT_BUSINESS) -> pd.DataFrame:
    """Clients triés par score décroissant : capture cumulée des partants, lift, gain net cumulé.

    Répond à « si je contacte les k premiers clients du classement, que se passe-t-il ? ».
    """
    y, score, mc = _as_arrays(y_true, score, monthly_charges)
    order = np.argsort(-score, kind="stable")
    y, mc = y[order], mc[order]
    n = np.arange(1, len(y) + 1)
    captured = np.cumsum(y)
    tp_value = np.cumsum(y * params.success_rate * params.clv_months * mc)
    return pd.DataFrame({
        "share_targeted": n / len(y),
        "share_churners_captured": captured / y.sum(),
        "lift": (captured / n) / y.mean(),
        "net_gain": tp_value - params.action_cost * n,
    })


def lift_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Table de lift par décile de risque (décile 1 = les 10 % de clients les plus risqués)."""
    y, proba = _as_arrays(y_true, proba)
    rank = pd.Series(proba).rank(method="first", ascending=False)
    decile = pd.qcut(rank, n_bins, labels=range(1, n_bins + 1))
    table = pd.DataFrame({"y": y, "proba": proba, "decile": decile}).groupby("decile", observed=True).agg(
        clients=("y", "size"), churners=("y", "sum"),
        predicted_rate=("proba", "mean"), churn_rate=("y", "mean"))
    table["lift"] = table["churn_rate"] / y.mean()
    table["cum_share_churners"] = table["churners"].cumsum() / y.sum()
    return table
