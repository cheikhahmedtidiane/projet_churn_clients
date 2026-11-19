"""Audit d'équité : performance, calibration et décisions du modèle par groupe de clients.

Être contacté signifie recevoir une offre de rétention : c'est un avantage pour le client.
Le risque d'injustice est donc d'être *oublié* (risque sous-estimé pour un groupe).
Le critère principal retenu est la calibration par groupe : un risque annoncé de 30 %
doit correspondre à 30 % de départs dans chaque groupe.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def group_report(y_true, proba, contacted, groups, n_boot: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Indicateurs par groupe, avec un intervalle bootstrap (95 %) sur l'écart de calibration.

    Colonnes :
        n, churn_rate (observé), mean_pred (risque moyen prédit), calibration_gap (prédit − observé)
        et son IC 95 %, pr_auc, roc_auc, contact_rate, recall (part des partants contactés),
        false_positive_rate (part des fidèles contactés).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    c = np.asarray(contacted, dtype=bool)
    g = np.asarray(groups).astype(str)
    rng = np.random.default_rng(random_state)
    rows = {}
    for value in pd.unique(g):
        k = g == value
        yk, pk, ck = y[k], p[k], c[k]
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, k.sum(), k.sum())
            boot.append(pk[idx].mean() - yk[idx].mean())
        both_classes = 0 < yk.sum() < len(yk)
        rows[value] = {
            "n": int(k.sum()),
            "churn_rate": yk.mean(),
            "mean_pred": pk.mean(),
            "calibration_gap": pk.mean() - yk.mean(),
            "gap_ci_low": np.percentile(boot, 2.5),
            "gap_ci_high": np.percentile(boot, 97.5),
            "pr_auc": average_precision_score(yk, pk) if both_classes else np.nan,
            "roc_auc": roc_auc_score(yk, pk) if both_classes else np.nan,
            "contact_rate": ck.mean(),
            "recall": ck[yk == 1].mean() if yk.any() else np.nan,
            "false_positive_rate": ck[yk == 0].mean() if (yk == 0).any() else np.nan,
        }
    return pd.DataFrame(rows).T.astype(float).assign(n=lambda d: d["n"].astype(int))
