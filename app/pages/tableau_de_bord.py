import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from churn.business import campaign_gain, expected_gain, targeting_curve
from churn.config import BusinessParams
from churn.viz import AQUA, BLUE, GRID, TEXT, TEXT_SECONDARY
from common import business_params, fmt_pct, fmt_usd, get_service

service = get_service()
params = business_params()

st.title("Tableau de bord de la campagne")
st.caption("Gain **réellement obtenu** sur 1 409 clients jamais vus par le modèle (jeu de test, issue connue), "
           "selon les hypothèses de la barre latérale.")

X, y = service.X_test, service.y_test.to_numpy()
proba = service.predict_proba(X)
mc = X["MonthlyCharges"].to_numpy(dtype=float)


@st.cache_data
def strategies(cost: float, rate: float, months: int) -> pd.DataFrame:
    p = BusinessParams(clv_months=months, action_cost=cost, success_rate=rate)
    rules = {
        "Ne contacter personne": np.zeros(len(y), dtype=bool),
        "Contacter tout le monde": np.ones(len(y), dtype=bool),
        "Seuil 0,5 (par défaut)": proba >= 0.5,
        "Gain espéré > 0 (modèle)": expected_gain(proba, mc, p) > 0,
        "Oracle (plafond théorique)": y == 1,
    }
    return pd.DataFrame({k: campaign_gain(y, t, mc, p) for k, t in rules.items()}).T


res = strategies(params.action_cost, params.success_rate, params.clv_months)
rule = res.loc["Gain espéré > 0 (modèle)"]
oracle = res.loc["Oracle (plafond théorique)", "net_gain"]
everyone = res.loc["Contacter tout le monde", "net_gain"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Gain net de la campagne", fmt_usd(rule["net_gain"], signed=True),
          help=f"{int(rule['n_targeted'])} clients contactés sur {len(y)}.")
c2.metric("Gain pour 1 000 clients", fmt_usd(rule["net_gain"] / len(y) * 1000))
c3.metric("vs « contacter tout le monde »", fmt_usd(rule["net_gain"] - everyone, signed=True))
c4.metric("Part du plafond théorique", fmt_pct(rule["net_gain"] / oracle, 0) if oracle > 0 else "—",
          help="Rapport au gain d'un devin qui contacterait exactement les partants.")

st.markdown("**Gain net selon la stratégie de ciblage**")
r = res.iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 3.2))
colors = [GRID if "Oracle" in n else AQUA if "modèle" in n else BLUE if "Seuil" in n else TEXT_SECONDARY
          for n in r.index]
ax.barh(r.index, r["net_gain"] / 1e3, color=colors, height=0.6)
span = max(r["net_gain"].max(), 1) / 1e3
for i, v in enumerate(r["net_gain"]):
    ax.text(v / 1e3 + (span * 0.02 if v >= 0 else -span * 0.02), i, fmt_usd(v), va="center",
            ha="left" if v >= 0 else "right", fontsize=9, color=TEXT)
lo = min(0, r["net_gain"].min() / 1e3 * 1.5)
ax.set_xlim(lo, span * 1.45)
ax.axvline(0, color=TEXT_SECONDARY, lw=1)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f} k$")
ax.grid(axis="y", visible=False)
st.pyplot(fig, width="stretch")
plt.close(fig)

st.markdown("**Gain net selon le nombre de clients contactés**")
curve = targeting_curve(y, expected_gain(proba, mc, params), mc, params)
k = int(rule["n_targeted"])
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(curve["share_targeted"], curve["net_gain"] / 1e3, color=BLUE, lw=2)
ax.axhline(0, color=TEXT_SECONDARY, lw=1)
if k > 0:
    g = curve["net_gain"].iloc[k - 1] / 1e3
    ax.plot(curve["share_targeted"].iloc[k - 1], g, "o", color=AQUA, markersize=8)
    ax.annotate(f"règle du modèle\n{k} clients ({fmt_pct(k / len(y), 0)})",
                (curve["share_targeted"].iloc[k - 1], g), xytext=(10, -30), textcoords="offset points",
                fontsize=8, color=TEXT)
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(lambda v, _: fmt_pct(v, 0))
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f} k$")
ax.set_xlabel("Part des clients contactés (par gain espéré décroissant)")
st.pyplot(fig, width="stretch")
plt.close(fig)

st.markdown(
    f"**Lecture.** Avec ces hypothèses, la règle du modèle contacte **{int(rule['n_targeted'])} clients** "
    f"(précision {fmt_pct(rule['precision'], 0)} : part de vrais partants parmi eux ; rappel "
    f"{fmt_pct(rule['recall'], 0)} : part des partants atteints) pour un coût de {fmt_usd(rule['cost'])}. "
    "Modifiez le coût ou le taux de succès dans la barre latérale : la règle s'adapte d'elle-même, "
    "et ne contacte jamais un client dont l'appel ferait perdre de l'argent en espérance."
)

with st.expander("Qualité du modèle sur le jeu de test", icon=":material/verified:"):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PR-AUC", f"{average_precision_score(y, proba):.3f}".replace(".", ","), help="Hasard : 0,265.")
    m2.metric("ROC-AUC", f"{roc_auc_score(y, proba):.3f}".replace(".", ","))
    m3.metric("Brier", f"{brier_score_loss(y, proba):.3f}".replace(".", ","), help="Plus bas = mieux.")
    m4.metric("Risque moyen prédit / observé", f"{fmt_pct(proba.mean())} / {fmt_pct(y.mean())}",
              help="Calibration : les deux doivent être proches.")
    st.caption(f"Modèle : {service.card['model']} entraîné sur {service.card['n_train']} clients "
               f"le {service.card['created_at'][:10]}. Choix et validation détaillés dans les notebooks 05 et 06.")
