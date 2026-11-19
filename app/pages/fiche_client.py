import matplotlib.pyplot as plt
import streamlit as st

from churn.viz import plot_explanation
from common import business_params, client_form, fmt_pct, fmt_usd, get_service

service = get_service()
params = business_params()

st.title("Fiche client")
st.caption("Risque de départ, décision de contact et explication du modèle pour un client.")

mode = st.radio("Quel client ?", ["Client existant", "Nouveau client"], horizontal=True,
                help="Les clients existants proviennent du jeu de test : le modèle ne les a jamais vus.")
truth = None
if mode == "Client existant":
    ids = list(service.X_test.index)
    default = ids.index("3244-DCJWY") if "3244-DCJWY" in ids else 0
    cid = st.selectbox("Identifiant client", ids, index=default)
    x = service.X_test.loc[[cid]]
    truth = int(service.y_test[cid])
else:
    raw = client_form(key="fiche")
    if raw is not None:
        st.session_state["new_client"] = service.prepare(raw)
    x = st.session_state.get("new_client")
    if x is None:
        st.info("Renseignez le client puis cliquez sur « Évaluer ce client ».")
        st.stop()
st.session_state["client"] = x

score = service.score(x, params).iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Risque de départ", fmt_pct(score["risque"]))
c2.metric("Seuil rentable", fmt_pct(score["seuil_rentabilite"]),
          help="Risque à partir duquel contacter ce client est rentable, compte tenu de sa facture.")
c3.metric("Gain espéré", fmt_usd(score["gain_espere"], signed=True), help="Gain espéré si l'on contacte ce client.")
c4.metric("Contacter ?", "Oui" if score["contacter"] else "Non")

if score["contacter"]:
    st.success(f"**Contacter ce client.** Son risque ({fmt_pct(score['risque'])}) dépasse son seuil de "
               f"rentabilité ({fmt_pct(score['seuil_rentabilite'])}) : l'appel rapporte "
               f"{fmt_usd(score['gain_espere'])} en espérance.")
else:
    st.info(f"**Ne pas contacter.** Son risque ({fmt_pct(score['risque'])}) est sous son seuil de "
            f"rentabilité ({fmt_pct(score['seuil_rentabilite'])}) : l'appel coûterait plus qu'il ne rapporte.")
if truth is not None:
    st.caption(f"Pour la démonstration : ce client est en réalité **{'parti' if truth else 'resté'}**.")

st.subheader("Pourquoi ce niveau de risque ?")
table = service.explain(x)
fig, ax = plt.subplots(figsize=(10, 4.6))
plot_explanation(ax, table, service.explainer.base_value)
st.pyplot(fig, width="stretch")
plt.close(fig)


def bullet(r):
    lever = " — *levier actionnable*" if r["actionnable"] else ""
    return f"- **{r['libellé']}** : {r['valeur']}{lever}"


left, right = st.columns(2)
risk = table[table["contribution"] > 0.05].head(4)
protect = table[table["contribution"] < -0.05].head(4)
left.markdown("**Ce qui augmente le risque**")
left.markdown("\n".join(bullet(r) for _, r in risk.iterrows()) or "_Aucun facteur notable._")
right.markdown("**Ce qui le protège**")
right.markdown("\n".join(bullet(r) for _, r in protect.iterrows()) or "_Aucun facteur notable._")
st.caption("Orange : pousse le risque vers le haut ; bleu : vers le bas, par rapport à un client de "
           "référence. Les leviers actionnables peuvent être testés dans **Simulation de leviers**.")
