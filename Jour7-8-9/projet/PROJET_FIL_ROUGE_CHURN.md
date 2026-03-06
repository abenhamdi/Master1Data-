# PROJET FIL ROUGE - SYSTÈME DE PRÉDICTION DE CHURN CLIENT

**Master 1 Data Science - Concepts et Technologies IA**

---

## INFORMATIONS GÉNÉRALES

**Durée totale**: 3 jours (21 heures)
- Jour 1: 7h - Exploration et Feature Engineering
- Jour 2: 7h - Modélisation et Optimisation
- Jour 3: 7h - Production et MLOps

**Modalité**: Projet autonome (individuel ou binôme)

**Niveau de difficulté**: Expert

**Livrables**:
1. 3 notebooks Jupyter complétés et documentés
2. Code source Python modulaire (preprocessing, predict, api)
3. API REST déployable (Flask ou FastAPI)
4. Dockerfile + docker-compose.yml
5. Rapport de synthèse (format Markdown ou PDF, 5-10 pages)
6. Présentation orale (15 minutes + 5 minutes Q&A)

---

## CONTEXTE MÉTIER

Vous êtes Data Scientist chez **TelecomPro**, un opérateur télécoms avec 7000+ clients. Le taux de churn annuel atteint 26.5%, représentant une perte de revenus significative.

**Objectif stratégique**: Développer un système de ML en production pour prédire le risque de churn et permettre des actions de rétention ciblées.

**Contraintes business**:
- Précision minimale: Recall ≥ 75% (détecter au moins 75% des churners)
- Déploiement: API REST temps réel (<200ms par prédiction)
- Monitoring: Détection de drift et A/B testing
- Explicabilité: Justifier les prédictions (SHAP)

---

## DATASET

**Source**: Kaggle - `mnassrib/telecom-churn-prediction`

**Caractéristiques**:
- 7043 observations, 21 variables (20 features + 1 target)
- Target: `Churn` (Yes/No)
- Déséquilibre: 73.5% No Churn / 26.5% Churn (ratio 2.77:1)
- Features: mix numériques (3) et catégorielles (17)

**Téléchargement**:
```bash
cd projet/data
python download_data.py
```

---

## JOUR 1 - EXPLORATION ET FEATURE ENGINEERING (7h)

### Objectifs du Jour 1
- Maîtriser le dataset et identifier les patterns de churn
- Créer un pipeline de preprocessing robuste
- Générer des features avancées (minimum 10 nouvelles features)
- Établir une baseline de référence

### Notebook: `01_EDA_Feature_Engineering.ipynb`

#### Partie 1.1 - Exploration Approfondie (2h)

**Mission 1.1.1 - Analyse Univariée**
- Analyser les distributions de toutes les variables
- Identifier les valeurs manquantes et aberrantes
- Analyser les corrélations avec la target

**Mission 1.1.2 - Analyse Bivariée**
- Taux de churn par segment (gender, contract, internet service, etc.)
- Impact de la tenure sur le churn
- Relation MonthlyCharges vs TotalCharges vs Churn

**Mission 1.1.3 - Insights Métier**
- Identifier les 5 profils clients les plus à risque
- Quantifier l'impact financier du churn (perte mensuelle moyenne)
- Formuler 3 hypothèses métier à tester

#### Partie 1.2 - Feature Engineering Avancé (3h)

**Mission 1.2.1 - Créer au minimum 10 nouvelles features**

Les features doivent inclure au moins :
- **Ratio et relations**: 
  - Ratio TotalCharges / tenure (dépense moyenne par mois)
  - Ratio MonthlyCharges / nombre de services
  
- **Agrégations**: 
  - Nombre total de services souscrits
  - Score d'engagement client
  
- **Interactions**: 
  - Combinaisons pertinentes (ex: Contract × InternetService)
  
- **Transformations mathématiques**: 
  - Log des charges, transformations Box-Cox si nécessaire
  
- **Encodages intelligents**: 
  - Target encoding pour variables catégorielles à haute cardinalité

**Mission 1.2.2 - Pipeline de Preprocessing**
Créer un pipeline Scikit-learn incluant:
- Gestion des valeurs manquantes
- Encodage des catégorielles (OneHot, Ordinal, Target selon pertinence)
- Normalisation/Standardisation
- Feature engineering automatisé

#### Partie 1.3 - Baseline et Validation (2h)

**Mission 1.3.1 - Modèle Baseline**
- Train/Test split stratifié (80/20)
- Logistic Regression simple
- Métriques: Accuracy, Precision, Recall, F1, ROC-AUC

**Mission 1.3.2 - Stratégie de Validation**
- Implémenter StratifiedKFold (5 folds)
- Analyser la stabilité des métriques
- Détecter l'overfitting

**Livrable Jour 1**:
- Notebook `01_EDA_Feature_Engineering.ipynb` complet
- Module `src/preprocessing.py` avec pipeline réutilisable
- Dataset enrichi sauvegardé

---

## JOUR 2 - MODÉLISATION ET OPTIMISATION (7h)

### Objectifs du Jour 2
- Comparer au minimum 5 algorithmes différents
- Optimiser les hyperparamètres
- Gérer le déséquilibre des classes
- Atteindre les objectifs business (Recall ≥ 75%)

### Notebook: `02_Modeling_Optimization.ipynb`

#### Partie 2.1 - Benchmark de Modèles (2h30)

**Mission 2.1.1 - Comparer au minimum 5-6 algorithmes**

Modèles obligatoires:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting (XGBoost ou LightGBM)
5. Support Vector Machine (SVM)
6. Au choix: KNN, CatBoost, Neural Network, etc.

Pour chaque modèle:
- Entraîner avec paramètres par défaut
- Cross-validation (5 folds)
- Métriques complètes: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Matrice de confusion
- Courbe ROC et Precision-Recall

**Mission 2.1.2 - Analyse Comparative**
- Tableau récapitulatif des performances
- Identifier les 3 meilleurs modèles
- Analyser les compromis Precision/Recall
- Justifier mathématiquement les résultats (pourquoi tel modèle performe mieux ?)

#### Partie 2.2 - Gestion du Déséquilibre (1h30)

**Mission 2.2.1 - Stratégies de Rééquilibrage**

Comparer au moins 3 stratégies:
1. **class_weight='balanced'** (Logistic Regression, Random Forest)
2. **SMOTE** (Synthetic Minority Over-sampling Technique)
   - Principe mathématique: Générer des échantillons synthétiques
   - Formule: `x_new = x_i + λ × (x_neighbor - x_i)` où λ ∈ [0,1]
3. **Undersampling** (RandomUnderSampler)
4. **Hybrid** (SMOTE + ENN ou Tomek Links)

**Mission 2.2.2 - Impact sur les Métriques**
- Comparer Recall, Precision, F1 avant/après rééquilibrage
- Analyser le compromis faux positifs / faux négatifs
- Sélectionner la stratégie optimale pour le business

#### Partie 2.3 - Optimisation Avancée (2h)

**Mission 2.3.1 - GridSearchCV ou RandomizedSearchCV**

Pour les 2 meilleurs modèles, optimiser:

**Random Forest**:
```python
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', None]
}
```

**XGBoost/LightGBM**:
- learning_rate: [0.01, 0.05, 0.1, 0.3]
- n_estimators: [100, 200, 500, 1000]
- max_depth: [3, 5, 7, 10]
- subsample: [0.6, 0.8, 1.0]
- colsample_bytree: [0.6, 0.8, 1.0]
- scale_pos_weight: ratio négatifs/positifs (pour XGBoost)

**Calcul du scale_pos_weight**:
```python
scale_pos_weight = (count_class_0) / (count_class_1)
# Pour notre dataset: ~2.77
```

**Mission 2.3.2 - Probabilité RandomizedSearchCV**

Si vous utilisez RandomizedSearchCV avec `n_iter=n`:
- Calculer la probabilité de trouver les paramètres dans le top 5% :
  
```
P(top 5%) = 1 - (0.95)^n
```

Exemple: n=100 → P ≈ 99.4%

**Mission 2.3.3 - Validation Finale**
- Sélectionner le meilleur modèle optimisé
- Évaluer sur le test set (jamais touché)
- Vérifier l'atteinte des objectifs: Recall ≥ 75%

#### Partie 2.4 - Analyse Avancée (1h)

**Mission 2.4.1 - Learning Curves**
- Analyser overfitting/underfitting
- Déterminer si plus de données améliorerait le modèle

**Mission 2.4.2 - Calibration des Probabilités**
- Vérifier la calibration (Platt Scaling)
- Formule Platt Scaling:

```
P_calibrated = 1 / (1 + exp(A × log(p/(1-p)) + B))
```

où A et B sont appris par régression logistique sur les probabilités brutes.

**Livrable Jour 2**:
- Notebook `02_Modeling_Optimization.ipynb` complet
- Meilleur modèle sauvegardé (`models/best_model.pkl`)
- Rapport comparatif des 5-6 modèles

---

## JOUR 3 - PRODUCTION ET MLOPS (7h)

### Objectifs du Jour 3
- Rendre le modèle explicable (SHAP, Feature Importance)
- Développer une API REST (Flask ou FastAPI)
- Conteneuriser avec Docker
- Implémenter monitoring basique et avancé (MLflow, drift detection)
- Préparer la présentation finale

### Notebook: `03_Production_MLOps.ipynb`

#### Partie 3.1 - Explicabilité (2h)

**Mission 3.1.1 - Feature Importance (MDI)**
- Mean Decrease Impurity pour Random Forest/XGBoost
- Top 10 features les plus importantes

**Mission 3.1.2 - Permutation Importance**
- Principe: Mesurer la dégradation de performance si feature permutée
- Formule:

```
PI(f) = S_baseline - S_permuted
```

où S est le score (ex: Recall) sur test set.

**Mission 3.1.3 - SHAP Values (Explicabilité Avancée)**

Implémenter SHAP pour expliquer les prédictions individuelles.

**Principe mathématique**: Valeurs de Shapley issues de la théorie des jeux coopératifs.

Formule SHAP:
```
φ_i = Σ_{S ⊆ N\{i}} [|S|!(n-|S|-1)!/n!] × [f(S∪{i}) - f(S)]
```

où:
- φ_i: Contribution de la feature i
- S: Sous-ensemble de features
- N: Ensemble de toutes les features
- f(S): Prédiction avec features S

**Livrables SHAP**:
- Summary plot (importance globale)
- Force plot (explication d'une prédiction individuelle)
- Dependence plot (impact d'une feature)
- Waterfall plot (décomposition d'une prédiction)

#### Partie 3.2 - API REST (2h30)

**Mission 3.2.1 - Créer une API avec Flask ou FastAPI**

**Structure de l'API**:
```
api/
├── app.py              # Point d'entrée
├── config.py           # Configuration
├── routes/
│   ├── predict.py      # Endpoint prédiction
│   └── health.py       # Health check
├── models/
│   └── best_model.pkl  # Modèle chargé
└── utils/
    └── preprocessing.py # Pipeline
```

**Endpoints obligatoires**:

1. `GET /health` - Health check
   - Retour: `{"status": "healthy", "model_loaded": true, "version": "1.0.0"}`

2. `POST /predict/single` - Prédiction individuelle
   - Input: JSON avec features d'un client
   - Output: `{"churn_probability": 0.78, "churn_prediction": "Yes", "risk_level": "High", "explanation": {...}}`

3. `POST /predict/batch` - Prédiction batch
   - Input: JSON array de clients
   - Output: Liste de prédictions

4. `GET /model/info` - Informations modèle
   - Retour: Métriques, date d'entraînement, features

**Mission 3.2.2 - Tests de l'API**
- Créer `tests/test_api.py` avec pytest
- Tester tous les endpoints
- Vérifier les temps de réponse (<200ms)

**Mission 3.2.3 - Documentation API**
- Si FastAPI: Documentation Swagger automatique
- Si Flask: Créer README_API.md avec exemples curl

#### Partie 3.3 - Conteneurisation Docker (1h30)

**Mission 3.3.1 - Créer Dockerfile**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Installation dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY . .

# Exposition du port
EXPOSE 5000

# Commande de lancement
CMD ["python", "api/app.py"]
```

**Mission 3.3.2 - Docker Compose**

Créer `docker-compose.yml` avec:
- Service API
- Service MLflow (tracking)
- Service Prometheus (monitoring)
- Volumes persistants

**Mission 3.3.3 - Build et Test**
```bash
docker build -t churn-api:latest .
docker run -p 5000:5000 churn-api:latest
# Tester l'API sur localhost:5000
```

#### Partie 3.4 - Monitoring et MLOps (1h)

**Mission 3.4.1 - Tracking avec MLflow**
- Logger les expériences (paramètres, métriques, modèles)
- Créer un run pour chaque entraînement
- Comparer les runs dans MLflow UI

**Mission 3.4.2 - Détection de Drift**

Implémenter une fonction de détection de concept drift:

**Méthodes**:
1. **Population Stability Index (PSI)**:
```
PSI = Σ (actual_i - expected_i) × ln(actual_i / expected_i)
```

Interprétation:
- PSI < 0.1: Pas de drift
- 0.1 ≤ PSI < 0.2: Drift modéré
- PSI ≥ 0.2: Drift significatif → ré-entraînement nécessaire

2. **KS Test** (Kolmogorov-Smirnov) pour variables continues

**Mission 3.4.3 - A/B Testing (Réflexion)**
- Concevoir un plan d'A/B testing pour comparer 2 modèles en production
- Définir les métriques de succès
- Calculer la taille d'échantillon nécessaire

**Mission 3.4.4 - Monitoring API**
- Métriques Prometheus: nombre de requêtes, latence, erreurs
- Dashboard Grafana (optionnel)

---

## RAPPORT DE SYNTHÈSE (2 pages minimum)

Le rapport doit inclure:

### 1. Executive Summary
- Objectif du projet
- Résultats clés (meilleur modèle, performances)
- Recommandations business

### 2. Méthodologie
- Pipeline de preprocessing
- Stratégie de feature engineering (10+ features créées)
- Approche de modélisation (5-6 modèles comparés)

### 3. Résultats
- Tableau comparatif des 5-6 modèles
- Métriques du modèle final (Recall ≥ 75% ?)
- Top 10 features les plus importantes

### 4. Explicabilité
- Interprétation SHAP
- Profils de clients à risque

### 5. Production
- Architecture de l'API
- Stratégie de monitoring
- Plan de déploiement

### 6. Limites et Améliorations
- Biais potentiels
- Pistes d'amélioration
- Prochaines étapes

---

## PRÉSENTATION ORALE (15 min + 5 min Q&A)

### Structure recommandée:

**Slide 1**: Titre et contexte (1 min)
**Slides 2-3**: Exploration du dataset et insights (2 min)
**Slide 4**: Feature engineering (2 min)
**Slides 5-6**: Benchmark de modèles (3 min)
- Tableau comparatif des 5-6 modèles
- Justification du choix final
**Slide 7**: Gestion du déséquilibre (2 min)
**Slide 8**: Explicabilité (SHAP) (2 min)
**Slide 9**: Architecture API et MLOps (2 min)
**Slide 10**: Conclusion et recommandations (1 min)

**Démo live**: Montrer l'API en action (prédiction d'un client)

---

## CRITÈRES D'ÉVALUATION

Voir grille détaillée dans `professeur/grille_evaluation/GRILLE_EVALUATION.md`

**Répartition**:
- Exploration et Feature Engineering: 20%
- Modélisation et Benchmark (5-6 modèles): 25%
- Optimisation et Gestion du déséquilibre: 15%
- Explicabilité (SHAP): 10%
- API REST et Production: 15%
- Monitoring et MLOps: 10%
- Rapport et Présentation: 5%

**Attendu pour l'excellence**:
- Recall ≥ 80% sur test set
- Au moins 15 features créées
- 6 modèles comparés avec justifications mathématiques
- API fonctionnelle avec tests unitaires
- Docker opérationnel
- SHAP correctement implémenté et interprété
- Monitoring drift implémenté

---

## RESSOURCES

### Documentation Officielle
- Scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- MLflow: https://mlflow.org/docs/
- FastAPI: https://fastapi.tiangolo.com/
- Flask: https://flask.palletsprojects.com/

### Aide-mémoire
- `AIDE_MEMOIRE.md`: Snippets de code réutilisables
- `GUIDE_METHODOLOGIQUE.md`: Méthodologie CRISP-DM appliquée

### Support
En cas de blocage, consultez:
1. Le formateur (aux jalons prévus)
2. La documentation officielle
3. Les notebooks d'exemples (si fournis)

---

## JALONS ET CHECKPOINTS

### Jour 1 - 16h00
Checkpoint: Présentation rapide (5 min) de l'EDA et des features créées

### Jour 2 - 16h00
Checkpoint: Résultats du benchmark de modèles (tableau comparatif)

### Jour 3 - 16h00
Présentation finale (15 min + 5 min Q&A)

---

## CONSEILS MÉTHODOLOGIQUES

1. **Commencez par l'exploration**: Ne sautez pas l'EDA, elle guide tout le reste
2. **Itérez**: Testez rapidement, analysez, améliorez
3. **Documentez en continu**: Commentaires, Markdown, docstrings
4. **Versionnez**: Git pour tracker vos expérimentations
5. **Testez l'API tôt**: Ne laissez pas la production pour le dernier moment
6. **Justifiez mathématiquement**: Expliquez pourquoi tel algorithme, tel hyperparamètre
7. **Pensez business**: Toujours relier vos choix techniques aux objectifs métier

---

**Bon courage et excellente montée en compétences !**

---

*Version 1.0 - Mars 2026*
