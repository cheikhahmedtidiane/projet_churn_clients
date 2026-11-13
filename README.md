# Prédiction du churn client — Télécom

> Identifier les clients à risque de départ, comprendre pourquoi, et décider lesquels cibler pour **maximiser le gain net** d'une campagne de rétention.

**Statut :** en cours de développement

## Le problème

Sur 7 043 clients d'un opérateur télécom, **26,5 % sont partis**, ce qui représente **~1,67 M$ de revenu annuel perdu**. Les clients qui partent paient en moyenne **21 % de plus** que ceux qui restent (74,4 $ contre 61,3 $ par mois).

L'objectif n'est pas seulement de prédire le churn, mais de répondre à une question métier :
**quels clients faut-il contacter, et est-ce rentable ?**

Cadrage détaillé : [docs/cadrage_metier.md](docs/cadrage_metier.md)

## Démarche

| Phase | Contenu | Statut |
|---|---|---|
| 0 | Cadrage métier | ✅ |
| 1 | Mise en place du projet | ✅ |
| 2 | Qualité et nettoyage des données | ✅ |
| 3 | Analyse exploratoire orientée insights | ⏳ |
| 4 | Feature engineering et pipeline | ⏳ |
| 5 | Modélisation et comparaison | ⏳ |
| 6 | Évaluation et optimisation business du seuil | ⏳ |
| 7 | Explicabilité (SHAP) | ⏳ |
| 8 | Application Streamlit | ⏳ |
| 9 | Valorisation et model card | ⏳ |

## Avancement et résultats clés

**Phase 2 — Qualité des données** ([notebook](notebooks/02_qualite_donnees.ipynb))
- 11 valeurs manquantes **cachées** dans `TotalCharges` (un espace `" "`, invisible pour `isna()`), correspondant toutes à des clients de **0 mois d'ancienneté** → fixées à 0 (nouveaux clients pas encore facturés) plutôt qu'imputées.
- Aucun doublon réel, aucune valeur aberrante, cohérence `TotalCharges ≈ ancienneté × facture` vérifiée.
- Nettoyage industrialisé dans [`src/churn/data.py`](src/churn/data.py) et couvert par des tests, réutilisé tel quel par l'application.

<img src="reports/figures/02_distribution_cible.png" width="480" alt="Distribution de la variable cible : 73,5 % restent, 26,5 % partent">

## Données

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) : 7 043 clients, 21 variables (profil, services souscrits, contrat, facturation, churn).

## Structure du projet

```
├── app/                 Application Streamlit
├── data/
│   ├── raw/             Données originales (jamais modifiées)
│   └── processed/       Données nettoyées (générées par le code)
├── docs/                Documentation (cadrage métier…)
├── models/              Modèles entraînés
├── notebooks/           Analyses numérotées par phase
├── reports/figures/     Graphiques exportés
├── src/churn/           Code réutilisable (config, nettoyage, features, modèle)
└── tests/               Tests unitaires (pytest)
```

## Installation

```bash
git clone https://github.com/cheikhahmedtidiane/projet_churn_clients.git
cd projet_churn_clients
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac : source .venv/bin/activate)
pip install -r requirements.txt
pytest                          # vérifie l'installation
```
