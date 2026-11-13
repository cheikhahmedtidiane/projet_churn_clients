"""Fonctions d'analyse exploratoire réutilisables (notebook et dashboard Streamlit)."""

import numpy as np
import pandas as pd
from scipy import stats

from churn.config import TARGET


def churn_rate_table(df: pd.DataFrame, col: str, target: str = TARGET) -> pd.DataFrame:
    """Taux de churn par modalité d'une variable.

    Colonnes retournées :
        clients      nombre de clients dans la modalité
        churners     nombre de clients partis
        churn_rate   part de clients partis dans la modalité
        share        part des clients totaux représentée par la modalité
        lift         churn_rate / taux de churn global (> 1 = segment plus à risque)
    """
    global_rate = df[target].mean()
    table = (
        df.groupby(col, observed=True)[target]
        .agg(clients="size", churners="sum", churn_rate="mean")
        .assign(
            share=lambda t: t["clients"] / t["clients"].sum(),
            lift=lambda t: t["churn_rate"] / global_rate,
        )
    )
    return table.sort_values("churn_rate", ascending=False)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér : force de l'association entre deux variables catégorielles (0 à 1)."""
    contingency = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(contingency, correction=False)[0]
    n = contingency.to_numpy().sum()
    k = min(contingency.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def categorical_association(df: pd.DataFrame, cols: list[str], target: str = TARGET,
                            alpha: float = 0.05) -> pd.DataFrame:
    """Test du χ² d'indépendance et V de Cramér de chaque variable avec la cible."""
    rows = []
    for col in cols:
        contingency = pd.crosstab(df[col], df[target])
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        rows.append({
            "variable": col,
            "chi2": chi2,
            "p_value": p_value,
            "cramers_v": cramers_v(df[col], df[target]),
        })
    out = pd.DataFrame(rows).set_index("variable")
    out["significatif"] = out["p_value"] < alpha
    return out.sort_values("cramers_v", ascending=False)


def numeric_association(df: pd.DataFrame, cols: list[str], target: str = TARGET,
                        alpha: float = 0.05) -> pd.DataFrame:
    """Test de Mann-Whitney (partants vs restants) et corrélation rang-bisériale.

    La corrélation rang-bisériale va de -1 à 1 :
        > 0  les partants ont des valeurs plus élevées
        < 0  les partants ont des valeurs plus faibles
    """
    rows = []
    churned, stayed = df[df[target] == 1], df[df[target] == 0]
    for col in cols:
        u_stat, p_value = stats.mannwhitneyu(churned[col], stayed[col], alternative="two-sided")
        rank_biserial = 2 * u_stat / (len(churned) * len(stayed)) - 1
        rows.append({
            "variable": col,
            "mediane_part": churned[col].median(),
            "mediane_reste": stayed[col].median(),
            "p_value": p_value,
            "rank_biserial": rank_biserial,
        })
    out = pd.DataFrame(rows).set_index("variable")
    out["significatif"] = out["p_value"] < alpha
    return out.reindex(out["rank_biserial"].abs().sort_values(ascending=False).index)
