import streamlit as st

from common import business_params, get_service

get_service()  # charge le modèle dès l'arrivée sur l'application
business_params()

st.title("Prédire le churn, décider qui contacter")
st.markdown(
    "Un opérateur télécom perd **26,5 %** de ses clients, soit **~1,67 M$** de revenu annuel. "
    "Cette application aide l'équipe de rétention à **identifier** les clients sur le départ, "
    "à **comprendre** pourquoi, et à **décider** qui contacter pour maximiser le gain net d'une campagne."
)

c1, c2, c3 = st.columns(3)
c1.metric("Gain net / 1 000 clients", "≈ 37 800 $", help="Règle du gain espéré, mesurée sur des clients jamais vus (jeu de test).")
c2.metric("vs « contacter tout le monde »", "× 1,9", help="53 311 $ contre 27 524 $ sur le jeu de test.")
c3.metric("PR-AUC (test)", "0,665", help="Qualité du classement des clients à risque. Hasard : 0,265. ROC-AUC : 0,848.")

st.subheader("Que pouvez-vous faire ici ?")
st.markdown(
    "- **Fiche client** : risque de départ d'un client, décision (contacter ou non) et **pourquoi**.\n"
    "- **Simulation de leviers** : effet d'une offre (contrat annuel, support technique…) sur le risque.\n"
    "- **Scoring d'un fichier** : importer une liste de clients et obtenir la **liste d'appels** priorisée.\n"
    "- **Tableau de bord** : gain de la campagne selon les hypothèses économiques."
)

st.subheader("Comment le modèle décide")
st.markdown(
    "Le modèle (régression logistique régularisée) estime une **probabilité de départ** `p`. "
    "Contacter un client rapporte en espérance `p × taux de succès × horizon × facture − coût de l'action`. "
    "On contacte **si ce gain espéré est positif** : un client qui paie cher mérite un appel même à risque modéré.\n\n"
    "Les hypothèses économiques sont réglables dans la **barre latérale** et s'appliquent à toutes les pages."
)
st.caption("Données : IBM Telco Customer Churn (7 043 clients). Projet complet, notebooks et code : "
           "[github.com/cheikhahmedtidiane/projet_churn_clients](https://github.com/cheikhahmedtidiane/projet_churn_clients)")
