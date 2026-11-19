# Model card : modèle de churn client (télécom)

> Fiche d'identité du modèle, au format des *model cards* (Mitchell et al., 2019) : usage prévu, données, performances, équité, limites et maintenance.

| | |
|---|---|
| **Nom** | `churn_model` ([`models/churn_model.joblib`](../models/churn_model.joblib), métadonnées : [`churn_model.json`](../models/churn_model.json)) |
| **Version** | 1.0, entraîné le 19/11/2026 |
| **Type** | Régression logistique régularisée (Elastic Net), dans un pipeline scikit-learn |
| **Sortie** | Probabilité qu'un client résilie (`predict_proba`), calibrée |
| **Décision associée** | Contacter le client si `p × 30 % × 12 × facture mensuelle > 50 $` (gain espéré positif) |
| **Environnement** | Python 3.12, scikit-learn 1.9.1, pandas 3.1.1 |
| **Démo** | [Application Streamlit](https://projetchurnclients-bcn6dfiavqhmbjycj87huf.streamlit.app/) |

## 1. Usage prévu

**Pour quoi faire.** Aider une équipe de rétention à **prioriser** les clients à contacter lors d'une campagne. Le modèle classe les clients par risque de départ. La règle de gain espéré décide qui contacter. Les explications SHAP aident le conseiller à choisir l'offre.

**Utilisateurs.** Équipe marketing et rétention (liste d'appels), conseillers (fiche client), direction (gain de campagne).

**Usages hors périmètre**
- **Décisions défavorables au client** : refus de service, hausse de prix, dégradation de l'offre. Le modèle a été conçu pour attribuer un **avantage** (une offre) ; l'utiliser pour pénaliser changerait totalement l'analyse d'équité.
- **Interprétation causale** : les effets estimés (ex. « un contrat 2 ans divise la cote par 4 ») sont des associations. Ils ne garantissent pas l'effet d'une action imposée.
- **Autres marchés ou périodes** sans revalidation : le modèle est appris sur un seul jeu de données historique.

## 2. Données

| | |
|---|---|
| **Source** | IBM Telco Customer Churn (Kaggle) : 7 043 clients, 21 variables, une seule date d'observation |
| **Cible** | `Churn` : le client a résilié dans le dernier mois (26,5 % de oui) |
| **Découpage** | 80 % entraînement (5 634) / 20 % test (1 409), stratifié, `random_state=42`. Le test est resté **sous clé** jusqu'à l'évaluation finale. |
| **Nettoyage** | 11 valeurs manquantes cachées dans `TotalCharges` (clients de 0 mois), mises à 0 ([notebook 02](../notebooks/02_qualite_donnees.ipynb)) |
| **Variables utilisées** | 17 variables d'origine (profil, services, contrat, facturation) + 2 features métier : `log_tenure`, `fiber_month_to_month` |
| **Variables retirées** | `gender` (aucun lien avec le churn), `PhoneService` (redondante), `TotalCharges` (≈ ancienneté × facture) |

## 3. Entraînement et sélection

- **Pipeline** : features métier → one-hot + standardisation → modèle. Un seul objet, sans fuite de données ([notebook 04](../notebooks/04_feature_engineering.ipynb)).
- **Comparaison** de 5 familles (Dummy, logistique, Random Forest, XGBoost, LightGBM) sur les mêmes 15 plis de validation croisée, t-test corrigé de Nadeau & Bengio ([notebook 05](../notebooks/05_modelisation.ipynb)).
- **Optimisation** des hyperparamètres avec Optuna, sur des plis distincts des plis d'évaluation. Résultat : `C ≈ 0,113`, `l1_ratio ≈ 0,211`, solveur `saga`.
- **Pas de rééquilibrage des classes** : `class_weight` et SMOTE n'amélioraient pas le classement et faussaient les probabilités.
- **Choix** : la logistique n'est pas significativement battue par XGBoost ou LightGBM (p = 0,48 et 0,31). Elle est retenue pour sa calibration, son explicabilité et sa simplicité.

## 4. Performances

| Métrique | Validation croisée (train) | Test (1 409 clients jamais vus) |
|---|---|---|
| **PR-AUC** (métrique principale) | 0,675 ± 0,017 | **0,665** (IC 95 % : 0,615 – 0,714) |
| ROC-AUC | 0,851 | 0,848 |
| Brier (plus bas = mieux) | 0,133 | 0,135 |
| Risque moyen prédit / observé | — | 26,8 % / 26,5 % |

Référence : un classement au hasard donne une PR-AUC de 0,265.

**Performance business** (test, hypothèses du cadrage) : la règle du gain espéré contacte 615 clients sur 1 409 et rapporte **53 311 $**, soit **67 %** du gain d'un oracle. C'est presque 2 fois plus que « contacter tout le monde » (27 524 $) et 33 % de plus que le seuil de 0,5 (40 087 $). Détails dans le [notebook 06](../notebooks/06_evaluation_business.ipynb).

## 5. Explicabilité

Les valeurs SHAP sont **exactes** pour ce modèle linéaire et regroupées par variable d'origine ([notebook 07](../notebooks/07_explicabilite.ipynb)).
- **Variables principales** : ancienneté, contrat et type d'internet, soit près de la moitié des décisions.
- **Leviers actionnables** : contrat, services de protection, paiement automatique.
- **Effets non interprétables** : la facture, le streaming et les lignes multiples. La facture est presque la somme du prix des services (R² = 0,999), ce qui rend leurs effets propres arbitraires.

## 6. Équité

L'audit complet est dans le [notebook 08](../notebooks/08_equite.ipynb). Être contacté = recevoir une offre : le risque d'injustice est d'être **oublié**. Le critère retenu est donc la **calibration par groupe**.

| Groupe | Calibration (prédit − observé) | ROC-AUC | Part contactée |
|---|---|---|---|
| Homme / Femme | +1,5 / −1,1 pt (non significatif) | 0,854 / 0,843 | 43 % / 45 % |
| Non senior / Senior | +0,7 / −2,5 pts (non significatif) | 0,849 / **0,784** | 38 % / 75 % |
| Sans / avec personnes à charge | +1,0 / −1,5 pt (non significatif) | 0,836 / 0,851 | 52 % / 24 % |

- **Aucun écart de calibration significatif.** Les écarts de taux de contact suivent des écarts réels de risque et de facture.
- **Point d'attention** : le classement est moins fin chez les seniors, sur un petit effectif (222 clients).
- **Attribut protégé** : le modèle utilise `SeniorCitizen`, mais son retrait ne coûte rien (PR-AUC 0,6750 sans, contre 0,6746 avec ; p = 0,78). **Recommandé pour la version 2.**

## 7. Limites et risques

- **Hypothèses économiques non mesurées.** Le coût de 50 $ et le taux de succès de 30 % viennent du cadrage. Le taux de succès est l'hypothèse la plus influente : il faut le **mesurer** par une campagne avec groupe témoin. La règle reste rentable dans tous les scénarios testés (coût de 25 à 100 $, succès de 10 à 50 %).
- **Données statiques.** Il n'y a qu'une seule date d'observation, donc aucune garantie de stabilité dans le temps (évolution des offres, de la concurrence, des prix).
- **Effet de l'action non modélisé.** Le modèle prédit le départ **sans** action. L'effet réel d'une offre sur chaque client (*uplift*) n'est pas estimé.
- **Taille de l'échantillon.** Sur 1 409 clients de test, les intervalles de confiance restent larges (PR-AUC ± 0,05).

## 8. Maintenance recommandée

| Quoi | Fréquence | Seuil d'alerte |
|---|---|---|
| Calibration globale (risque prédit vs départs observés) | Après chaque campagne | Écart > 3 points |
| PR-AUC sur les clients récents | Trimestrielle | < 0,60 |
| Calibration et taux de contact par groupe | Après chaque campagne | IC de l'écart de calibration excluant 0 |
| Dérive des variables (contrat, facture, ancienneté) | Mensuelle | Distribution nettement différente de l'entraînement |
| Taux de succès réel des actions (groupe témoin) | Chaque campagne | Mettre à jour `BusinessParams` |

**Version 2 envisagée** : retrait de `SeniorCitizen`, réentraînement sur des données plus récentes, puis modèle d'*uplift* une fois des données de campagne disponibles.
