import matplotlib.pyplot as plt
import streamlit as st

from churn.service import CATEGORY_OPTIONS
from churn.viz import BLUE, GRID, TEXT, TEXT_SECONDARY
from common import FIELD_LABELS, business_params, fmt_pct, fmt_usd, get_service, option_label

service = get_service()
params = business_params()

st.title("Simulation de leviers")
st.caption("Que devient le risque de départ si l'on propose une offre à ce client ?")

# --- Choix du client : celui de la fiche client, ou un client du jeu de test --------------
ids = list(service.X_test.index)
from_fiche = st.session_state.get("client")
choices = (["Client de la fiche client"] if from_fiche is not None else []) + ids
default = 0 if from_fiche is not None else (ids.index("3244-DCJWY") if "3244-DCJWY" in ids else 0)
choice = st.selectbox("Client", choices, index=default,
                      help="Par défaut, le dernier client consulté dans la page Fiche client.")
x = from_fiche if choice == "Client de la fiche client" else service.X_test.loc[[choice]]
row = x.iloc[0]
base = service.score(x, params).iloc[0]
st.markdown(f"Risque actuel : **{fmt_pct(base['risque'])}** · ancienneté {int(row['tenure'])} mois · "
            f"facture {row['MonthlyCharges']:.2f} $ · contrat {option_label('Contract', row['Contract'])}"
            .replace(".", ",", 1))

st.warning("Ces simulations reflètent des **associations** apprises sur l'historique, pas des effets causaux "
           "garantis : un client à qui l'on propose un contrat 2 ans ne se comportera pas forcément comme "
           "un client qui l'a choisi. Chaque levier est à valider par un test A/B.", icon=":material/science:")


def offer_ceiling(p_before: float, p_after: float) -> float:
    """Valeur de la baisse de risque : coût maximal d'une offre pour qu'elle reste rentable."""
    return (p_before - p_after) * params.clv_months * row["MonthlyCharges"]


# --- 1. Leviers classés -----------------------------------------------------------------
st.subheader("1. Leviers recommandés pour ce client")
levers = service.recommend_levers(x, params)
levers = levers[levers["baisse_relative"] >= 0.01]  # leviers sans effet notable masqués
if levers.empty:
    st.info("Aucun levier applicable : le client a déjà le contrat le plus long, les services de "
            "protection et un paiement automatique (ou n'a pas internet).")
else:
    levers["valeur"] = [offer_ceiling(a, b) for a, b in zip(levers["risque_avant"], levers["risque_apres"])]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(levers) + 1.2))
    y = range(len(levers))[::-1]
    for yi, (_, r) in zip(y, levers.iterrows()):
        ax.plot([r["risque_apres"], r["risque_avant"]], [yi, yi], color=GRID, lw=3, zorder=1)
        ax.plot(r["risque_avant"], yi, "o", color=TEXT_SECONDARY, markersize=8, zorder=2)
        ax.plot(r["risque_apres"], yi, "o", color=BLUE, markersize=9, zorder=3)
        ax.text(r["risque_apres"] - 0.01, yi, fmt_pct(r["risque_apres"]), ha="right", va="center",
                fontsize=9, color=TEXT)
    ax.set_yticks(list(y), levers["action"])
    ax.set_xlim(0, min(1, base["risque"] * 1.15 + 0.02))
    ax.xaxis.set_major_formatter(lambda v, _: fmt_pct(v, 0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Risque de départ : actuel (gris) → après le levier (bleu)")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    shown = levers[["action", "risque_avant", "risque_apres", "baisse_relative", "valeur"]].rename(columns={
        "action": "Levier", "risque_avant": "Risque actuel", "risque_apres": "Risque après",
        "baisse_relative": "Baisse relative", "valeur": "Coût max. de l'offre"})
    st.dataframe(shown.style.format({"Risque actuel": fmt_pct, "Risque après": fmt_pct,
                                     "Baisse relative": lambda v: fmt_pct(v, 0), "Coût max. de l'offre": fmt_usd}),
                 hide_index=True, width="stretch")
    st.caption("**Coût max. de l'offre** = baisse du risque × horizon × facture : au-delà, l'offre "
               "(remise, service offert…) coûte plus que la valeur qu'elle préserve en espérance.")

# --- 2. Simulation libre ------------------------------------------------------------------
st.subheader("2. Simulation libre")
has_internet = row["InternetService"] != "No"
changes = {}
cols = st.columns(3)
for i, col in enumerate(["Contract", "PaymentMethod", "OnlineSecurity", "TechSupport", "OnlineBackup",
                         "DeviceProtection"]):
    options = [o for o in CATEGORY_OPTIONS[col] if o != "No internet service"]
    internet_only = col not in ("Contract", "PaymentMethod")
    current = str(row[col])
    if internet_only and not has_internet:
        cols[i % 3].selectbox(FIELD_LABELS[col], ["Sans internet"], disabled=True, key=f"sim_{col}")
        continue
    value = cols[i % 3].selectbox(FIELD_LABELS[col], options, index=options.index(current),
                                  format_func=lambda v, c=col: option_label(c, v), key=f"sim_{col}_{choice}")
    if value != current:
        changes[col] = value

if changes:
    sim = service.simulate(x, changes, params)
    c1, c2, c3 = st.columns(3)
    c1.metric("Risque après simulation", fmt_pct(sim["risque_apres"]),
              delta=f"{(sim['risque_apres'] - sim['risque_avant']) * 100:+.1f} pts".replace(".", ","),
              delta_color="inverse")
    c2.metric("Baisse relative du risque", fmt_pct(1 - sim["risque_apres"] / sim["risque_avant"], 0))
    c3.metric("Coût max. de l'offre", fmt_usd(offer_ceiling(sim["risque_avant"], sim["risque_apres"])),
              help="Au-delà de ce coût, la combinaison d'offres n'est plus rentable en espérance.")
else:
    st.info("Modifiez un ou plusieurs leviers ci-dessus pour voir leur effet combiné.")
