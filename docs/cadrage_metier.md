# Phase 0 — Cadrage métier

## 1. Contexte

Un opérateur télécom (dataset public *IBM Telco Customer Churn*, 7 043 clients) constate qu'une part importante de ses abonnés résilie son contrat. Acquérir un nouveau client coûte en général **5 à 7 fois plus cher** que d'en retenir un existant : la rétention est donc un levier de rentabilité direct.

### Constat chiffré (données brutes)

| Indicateur | Valeur |
|---|---|
| Clients | 7 043 |
| Clients partis (churn) | 1 869 → **26,5 %** |
| Facture mensuelle moyenne (tous clients) | 64,8 $ |
| Facture mensuelle moyenne (clients partis) | **74,4 $** |
| Revenu mensuel perdu | ~139 k$ |
| Revenu perdu sur 12 mois | **~1,67 M$** |
| Churn — contrat mensuel / 1 an / 2 ans | 42,7 % / 11,3 % / 2,8 % |

Les clients qui partent sont **plus rentables que la moyenne** : le problème n'est pas marginal.

## 2. Problématique

> **Comment identifier à l'avance les clients à fort risque de départ, comprendre pourquoi ils partent, et décider lesquels cibler avec une offre de rétention pour maximiser le gain net de l'entreprise ?**

Elle se décompose en trois questions :

1. **Prédire** : quelle est la probabilité qu'un client donné résilie ? → *classification binaire supervisée*.
2. **Expliquer** : quels facteurs poussent un client à partir, globalement et individuellement ? → *explicabilité (SHAP)*.
3. **Décider** : à partir de quel niveau de risque est-il rentable de contacter un client ? → *optimisation du seuil sous contrainte de coût*.

## 3. Utilisateurs cibles

- **Équipe marketing / rétention** : obtient une liste priorisée de clients à contacter et l'action recommandée.
- **Conseiller client** : consulte le risque et les raisons pour un client précis (démo Streamlit).
- **Direction** : visualise le gain financier attendu d'une campagne.

## 4. Hypothèses économiques

Le dataset ne contient pas de données de coût : on pose des hypothèses **explicites et ajustables** (elles seront paramétrables dans l'application).

| Paramètre | Valeur par défaut | Justification |
|---|---|---|
| Valeur d'un client retenu (CLV) | 12 × facture mensuelle | Revenu préservé sur un horizon d'un an |
| Coût d'une action de rétention | 50 $ par client contacté | Appel + remise commerciale |
| Taux de succès de l'action | 30 % | Part des clients « à risque » contactés qui restent finalement |

### Matrice de gain (par rapport à « ne rien faire »)

| | Le client allait partir | Le client allait rester |
|---|---|---|
| **Prédit « part » → contacté** | Vrai positif : `0,30 × CLV − 50 $` | Faux positif : `− 50 $` (offre inutile) |
| **Prédit « reste » → non contacté** | Faux négatif : `0 $` (client perdu, comme sans modèle) | Vrai négatif : `0 $` |

Avec une CLV moyenne d'un partant d'environ 893 $ (12 × 74,4), un vrai positif rapporte ~218 $, alors qu'un faux positif ne coûte que 50 $.
**Conséquence : rater un partant coûte bien plus cher qu'une fausse alerte. On privilégiera le rappel (recall) et on ne gardera pas le seuil de 0,5 par défaut.**

## 5. Métriques

| Niveau | Métrique | Rôle |
|---|---|---|
| **Business (principale)** | Gain net de la campagne ($) | Critère final de choix du modèle et du seuil |
| Technique | PR-AUC (Average Precision) | Qualité du classement, adaptée aux classes déséquilibrées |
| Technique | Recall, Precision, F2-score | Au seuil retenu (F2 donne plus de poids au recall) |
| Technique | ROC-AUC | Comparaison avec la littérature |
| Écartée | Accuracy | Trompeuse : prédire « personne ne part » donne déjà 73,5 % |

## 6. Critères de succès

1. Le modèle final **bat la baseline** (régression logistique) en PR-AUC sur le jeu de test.
2. Le gain net de la campagne guidée par le modèle est **supérieur aux deux stratégies naïves** : « contacter tout le monde » et « ne contacter personne ».
3. Les prédictions sont **explicables** au niveau global et individuel.
4. Le tout est **utilisable** par un non-technicien via une application Streamlit.

## 7. Limites connues du dataset

- **Photo à un instant T** : pas d'historique temporel. On ne sait pas *quand* le client part, seulement s'il est parti.
- **Données anciennes et américaines** (prix en $) : les ordres de grandeur sont illustratifs.
- **Hypothèses de coût fictives** : d'où leur caractère paramétrable.
- **Variables sensibles** (`gender`, `SeniorCitizen`) : à surveiller pour éviter un ciblage discriminatoire (vérification de biais en phase 9).

## 8. Bilan en fin de projet (Phase 9)

Résultats mesurés sur le jeu de test (1 409 clients jamais vus), confrontés aux critères de succès de la section 6.

| # | Critère | Résultat | Statut |
|---|---|---|---|
| 1 | Le modèle final bat la baseline (régression logistique) en PR-AUC | Sans objet tel que formulé : la baseline **est** le modèle retenu. Les modèles plus complexes (XGBoost, LightGBM optimisés) ne l'ont pas battue significativement (p = 0,48 et 0,31). Le modèle bat largement le vrai plancher (classement au hasard : PR-AUC 0,265 → 0,665 sur le test). | ⚠️ Reformulé |
| 2 | Gain net supérieur aux stratégies naïves | **53 311 $** contre 27 524 $ (tout le monde) et 0 $ (personne), soit 67 % du gain d'un oracle. Écarts confirmés par bootstrap. | ✅ |
| 3 | Prédictions explicables, au niveau global et individuel | SHAP exact (modèle linéaire), regroupé par variable et traduit en phrases. Distinction entre leviers actionnables et simples corrélats. | ✅ |
| 4 | Utilisable par un non-technicien | Application Streamlit déployée : fiche client, simulation de leviers, scoring de fichier, tableau de bord. | ✅ |

**Leçon sur le critère n° 1.** Il supposait qu'un modèle plus complexe ferait mieux. Les données ont montré le contraire : le problème est presque linéaire, et la logistique offre en plus la meilleure calibration. Ce critère aurait dû être formulé ainsi : *« le modèle retenu n'est significativement battu par aucun autre, et il bat le classement au hasard »*. Ainsi formulé, il est atteint.

**Variables sensibles** (section 7) : l'audit d'équité ne montre aucun écart de calibration significatif selon le genre, l'âge ou les personnes à charge. Le retrait de `SeniorCitizen`, sans perte de performance, est recommandé pour une version 2 ([model card](model_card.md), [notebook 08](../notebooks/08_equite.ipynb)).

**Métriques prévues en section 5.** Le gain net a bien été le critère final de décision. Le F2-score n'a pas été nécessaire : la règle du gain espéré fixe directement le compromis entre précision et rappel à partir des coûts, plutôt que par une pondération arbitraire.
