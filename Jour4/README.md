# Jour 4 - TP Détection de Fraude Bancaire

**Master 1 Data Engineering - Concepts & Techno IA**  
**Durée : 7 heures**  
**Difficulté : Expert**

---

## Description

Ce TP complet porte sur la **détection de fraude bancaire** avec Machine Learning. Les étudiants développeront un système de détection optimisé capable d'identifier les transactions frauduleuses avec un Recall ≥ 0.85.

### Objectifs Pédagogiques

- Maîtriser le **Feature Engineering avancé** (15+ features)
- Gérer des **données fortement déséquilibrées** (0.17% de fraudes)
- Comparer **6 algorithmes** (LR, DT, RF, SVM, KNN, XGBoost)
- Optimiser avec **GridSearchCV et RandomizedSearchCV**
- Analyser l'**explicabilité** (MDI, Permutation Importance, SHAP Values)
- Implémenter une **API Flask** et des tests unitaires
- Déployer un modèle en production

---

## Structure du TP

```
Jour4/
├── SUJET_TP4_DETECTION_FRAUDE.md       # Sujet complet du TP
├── AIDE_MEMOIRE.md                     # Aide-mémoire technique
├── requirements.txt                    # Dépendances Python
├── TP4 - DÉTECTION DE FRAUDE BANCAIRE- M1.pdf  # Sujet en PDF
│
├── notebooks/
│   ├── TP4_Detection_Fraude_ETUDIANT.ipynb    # Notebook étudiant (140 cellules)
│   └── README_NOTEBOOK.md              # Instructions notebook
│
├── data/
│   ├── download_data.py                # Script de téléchargement dataset
│   └── README_DATA.md                  # Documentation dataset
│
├── utils/
│   ├── feature_engineering.py          # Fonctions de feature engineering
│   ├── predict.py                      # Classe FraudDetector
│   └── __init__.py
│
└── models/                             # Dossier pour sauvegarder les modèles
```

---

## Installation

### 1. Cloner le repository

```bash
git clone https://github.com/abenhamdi/Master1Data-.git
cd Master1Data-/Jour4
```

### 2. Créer un environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Télécharger le dataset

Le dataset provient de Kaggle : [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

**Option 1 : Téléchargement automatique (avec API Kaggle)**

```bash
cd data
python download_data.py
```

**Option 2 : Téléchargement manuel**

1. Télécharger `creditcard.csv` depuis Kaggle
2. Placer le fichier dans `data/creditcard.csv`

---

## Utilisation

### Lancer le notebook

```bash
jupyter notebook notebooks/TP4_Detection_Fraude_ETUDIANT.ipynb
```

### Structure du notebook (140 cellules)

**Partie 1 : Exploration & Feature Engineering (2h)**
- Analyse exploratoire
- Analyse temporelle approfondie
- Création de 15+ features (temporelles, polynomiales, ratios, écarts)

**Partie 2 : Modélisation Baseline (1h30)**
- 6 modèles : Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost
- Comparaison SMOTE vs class_weight
- Métriques : Precision, Recall, F1, PR-AUC

**Partie 3 : Optimisation Avancée (2h)**
- Pipeline ML complet
- GridSearchCV sur Random Forest
- RandomizedSearchCV (comparaison)
- Optimisation de XGBoost

**Partie 4 : Analyse & Diagnostic (1h)**
- Feature Importance (MDI)
- Permutation Importance
- SHAP Values (explicabilité avancée)
- Learning Curves
- Calibration des probabilités

**Partie 5 : Déploiement & Production (1h)**
- Validation temporelle (TimeSeriesSplit)
- Sérialisation du modèle
- API Flask
- Tests unitaires

---

## Dataset

**Source** : [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

**Caractéristiques** :
- 284,807 transactions
- 492 fraudes (0.172%)
- 30 features (Time, V1-V28 PCA, Amount, Class)
- Données anonymisées par PCA
- 2 jours de transactions

**Déséquilibre** : 1 fraude pour 600 transactions légitimes

---

## Concepts Clés

### Feature Engineering
- Encodage cyclique : `sin(2π × hour/24)`, `cos(2π × hour/24)`
- Transformations : log, sqrt, polynomiales
- Interactions : `Amount × V_i`
- Z-score : `|x - μ| / σ`

### Gestion du Déséquilibre
- `class_weight='balanced'`
- SMOTE : `x_new = x_i + λ × (x_neighbor - x_i)`
- `scale_pos_weight` pour XGBoost

### Optimisation
- GridSearchCV : recherche exhaustive
- RandomizedSearchCV : `P(top 5%) = 1 - (0.95)^n`

### Explicabilité
- MDI (Mean Decrease Impurity)
- Permutation Importance : `PI(f) = Score_baseline - Score_permuted`
- SHAP : `φ_i = Σ [|S|!(n-|S|-1)!/n!] × [f(S∪{i}) - f(S)]`

---

## Objectifs de Performance

### Minimums
- **Recall** : ≥ 0.80
- **Precision** : ≥ 0.70
- **F1-Score** : ≥ 0.75
- **PR-AUC** : ≥ 0.75

### Excellents
- **Recall** : ≥ 0.90
- **Precision** : ≥ 0.85
- **F1-Score** : ≥ 0.87
- **PR-AUC** : ≥ 0.85

---

## Ressources

### Documentation
- [Scikit-Learn](https://scikit-learn.org/)
- [Imbalanced-Learn](https://imbalanced-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)

### Articles Recommandés
- "Learning from Imbalanced Data" (He & Garcia, 2009)
- "SMOTE: Synthetic Minority Over-sampling Technique" (Chawla et al., 2002)

---

## Auteur

**BENHAMDI Ayoub**  
Master 1 Data Engineering - YNOV Montpellier  
Décembre 2025

---

## Licence

Ce matériel pédagogique est destiné aux étudiants du Master 1 Data Engineering de YNOV Montpellier.

