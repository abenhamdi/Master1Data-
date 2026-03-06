# GUIDE MÉTHODOLOGIQUE - PROJET CHURN PREDICTION

## CRISP-DM appliqué au Machine Learning

Ce guide détaille la méthodologie **CRISP-DM** (Cross-Industry Standard Process for Data Mining) appliquée à votre projet de prédiction de churn.

---

## 1. COMPRÉHENSION MÉTIER (Business Understanding)

### Objectifs Business
- Réduire le taux de churn de 26.5% à <20% sur 12 mois
- ROI attendu: Réduction des coûts d'acquisition (5x plus cher qu'une rétention)
- Identifier les clients à risque 2 mois avant le churn

### Questions Métier à Répondre
1. Quels sont les profils clients les plus susceptibles de partir ?
2. Quels facteurs ont le plus d'impact sur le churn ?
3. Quel est le moment optimal pour une action de rétention ?
4. Quel budget allouer aux campagnes de rétention ?

### Traduction en Objectifs ML
- **Métrique primaire**: Recall ≥ 75% (détecter 75% des churners)
- **Métrique secondaire**: Precision ≥ 60% (limiter les faux positifs)
- **Contrainte**: Latence API <200ms
- **Explicabilité**: Justifier chaque prédiction (SHAP)

---

## 2. COMPRÉHENSION DES DONNÉES (Data Understanding)

### Phase d'Exploration

#### Étape 2.1 - Collecte des Données
- Dataset Kaggle: 7043 clients, 21 variables
- Période couverte: Plusieurs années
- Granularité: 1 ligne = 1 client

#### Étape 2.2 - Description des Données
```python
# Checklist d'exploration
df.info()                  # Types, valeurs manquantes
df.describe()              # Statistiques descriptives
df['Churn'].value_counts() # Distribution de la target
df.isnull().sum()          # Valeurs manquantes par colonne
```

#### Étape 2.3 - Analyse de la Qualité
- **Valeurs manquantes**: TotalCharges (11 valeurs)
- **Valeurs aberrantes**: Clients avec tenure=0 mais TotalCharges>0
- **Incohérences**: TotalCharges ≠ tenure × MonthlyCharges pour certains clients
- **Duplicatas**: customerID uniques ?

#### Étape 2.4 - Analyse Univariée
Pour chaque variable:
- **Numériques**: Distribution (histogramme, boxplot, statistiques)
- **Catégorielles**: Fréquences, modalités, cardinalité

#### Étape 2.5 - Analyse Bivariée
- Taux de churn par catégorie:
```python
for col in categorical_features:
    print(df.groupby(col)['Churn'].value_counts(normalize=True))
```
- Corrélations numériques:
```python
df[numerical_features + ['Churn_binary']].corr()
```

#### Étape 2.6 - Insights Métier
Exemples de questions:
- Les clients avec contrat month-to-month churnent-ils plus ?
- Y a-t-il un seuil de tenure critique (ex: <6 mois) ?
- L'internet fiber optic a-t-il un impact sur le churn ?

---

## 3. PRÉPARATION DES DONNÉES (Data Preparation)

### Phase de Transformation

#### Étape 3.1 - Nettoyage
```python
# Gestion des valeurs manquantes
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)

# Suppression de customerID (non informatif)
df.drop('customerID', axis=1, inplace=True)
```

#### Étape 3.2 - Feature Engineering

##### 3.2.1 - Features de Ratio
```python
# Dépense moyenne par mois
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)

# Ratio charges mensuelles / tenure
df['ChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
```

##### 3.2.2 - Features d'Agrégation
```python
# Nombre total de services
service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                'StreamingTV', 'StreamingMovies']

df['TotalServices'] = (df[service_cols] == 'Yes').sum(axis=1)

# Score d'engagement
df['EngagementScore'] = df['tenure'] * df['TotalServices']
```

##### 3.2.3 - Features d'Interaction
```python
# Contract × InternetService
df['Contract_Internet'] = df['Contract'] + '_' + df['InternetService']

# PaymentMethod × PaperlessBilling
df['Payment_Paperless'] = df['PaymentMethod'] + '_' + df['PaperlessBilling']
```

##### 3.2.4 - Features Binaires
```python
# Client nouveau (<6 mois)
df['IsNewCustomer'] = (df['tenure'] < 6).astype(int)

# Client premium (>80$ par mois)
df['IsPremium'] = (df['MonthlyCharges'] > 80).astype(int)
```

##### 3.2.5 - Transformations Mathématiques
```python
# Log des charges (si distribution skewed)
df['LogMonthlyCharges'] = np.log1p(df['MonthlyCharges'])
df['LogTotalCharges'] = np.log1p(df['TotalCharges'])

# Polynomiales (si relation non-linéaire détectée)
df['Tenure_Squared'] = df['tenure'] ** 2
```

#### Étape 3.3 - Encodage

##### 3.3.1 - Target Encoding
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Churn_binary'] = le.fit_transform(df['Churn'])  # Yes=1, No=0
```

##### 3.3.2 - Variables Catégorielles

**OneHot Encoding** (cardinalité faible, <10 modalités):
```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded = ohe.fit_transform(df[['gender', 'Partner', 'Dependents']])
```

**Ordinal Encoding** (ordre naturel):
```python
from sklearn.preprocessing import OrdinalEncoder

# Contract: Month-to-month < One year < Two year
contract_order = [['Month-to-month', 'One year', 'Two year']]
oe = OrdinalEncoder(categories=contract_order)
df['Contract_encoded'] = oe.fit_transform(df[['Contract']])
```

**Target Encoding** (cardinalité élevée):
```python
# Moyenne de la target par modalité
target_mean = df.groupby('PaymentMethod')['Churn_binary'].mean()
df['PaymentMethod_encoded'] = df['PaymentMethod'].map(target_mean)
```

#### Étape 3.4 - Normalisation
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', ...]
df[numerical_features] = scaler.fit_transform(df[numerical_features])
```

#### Étape 3.5 - Split Train/Test
```python
from sklearn.model_selection import train_test_split

X = df.drop(['Churn', 'Churn_binary'], axis=1)
y = df['Churn_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.2%}")
print(f"Test churn rate: {y_test.mean():.2%}")
```

#### Étape 3.6 - Pipeline Scikit-learn
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Définir les transformations par type de colonne
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat_ohe', OneHotEncoder(handle_unknown='ignore'), categorical_low_card),
        ('cat_ord', OrdinalEncoder(), categorical_ordinal)
    ])

# Pipeline complet
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

---

## 4. MODÉLISATION (Modeling)

### Phase d'Entraînement

#### Étape 4.1 - Sélection des Algorithmes
Critères de choix:
- **Interprétabilité**: Logistic Regression, Decision Tree
- **Performance**: Random Forest, Gradient Boosting
- **Robustesse au déséquilibre**: XGBoost avec scale_pos_weight
- **Vitesse d'inférence**: Logistic Regression, KNN

#### Étape 4.2 - Baseline
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

#### Étape 4.3 - Benchmark de Modèles
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Recall': recall_score(y_test, model.predict(X_test)),
        'Precision': precision_score(y_test, model.predict(X_test)),
        'F1': f1_score(y_test, model.predict(X_test))
    }

pd.DataFrame(results).T.sort_values('ROC-AUC', ascending=False)
```

#### Étape 4.4 - Gestion du Déséquilibre

##### Méthode 1: class_weight
```python
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Poids automatiques
    random_state=42
)
rf_balanced.fit(X_train, y_train)
```

##### Méthode 2: SMOTE
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Avant SMOTE: {y_train.value_counts()}")
print(f"Après SMOTE: {y_train_smote.value_counts()}")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_smote, y_train_smote)
```

##### Méthode 3: scale_pos_weight (XGBoost)
```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

xgb_balanced = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False
)
xgb_balanced.fit(X_train, y_train)
```

#### Étape 4.5 - Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    rf_balanced, X_train, y_train,
    cv=skf, scoring='recall', n_jobs=-1
)

print(f"Recall CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

#### Étape 4.6 - Optimisation des Hyperparamètres

##### GridSearchCV
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Meilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score: {grid_search.best_score_:.4f}")
```

##### RandomizedSearchCV
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=100,  # P(top 5%) = 1 - 0.95^100 ≈ 99.4%
    cv=5,
    scoring='recall',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
```

---

## 5. ÉVALUATION (Evaluation)

### Phase de Validation

#### Étape 5.1 - Métriques de Classification

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_proba),
    'PR-AUC': average_precision_score(y_test, y_proba)
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

#### Étape 5.2 - Matrice de Confusion
```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédiction')
plt.ylabel('Réalité')
plt.title('Matrice de Confusion')
plt.show()

# Calcul des coûts business
TN, FP, FN, TP = cm.ravel()
cout_FN = 200  # Coût de perdre un client
cout_FP = 50   # Coût d'une action de rétention inutile
cout_total = FN * cout_FN + FP * cout_FP
print(f"Coût total: {cout_total}€")
```

#### Étape 5.3 - Courbes ROC et Precision-Recall
```python
from sklearn.metrics import roc_curve, precision_recall_curve

# ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc_score(y_test, y_proba):.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision, label=f'PR-AUC = {average_precision_score(y_test, y_proba):.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

#### Étape 5.4 - Optimisation du Seuil
```python
# Trouver le seuil optimal selon le critère business
best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.1, 0.9, 0.01):
    y_pred_threshold = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_threshold)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Seuil optimal: {best_threshold:.2f}")
print(f"F1-Score: {best_f1:.4f}")

# Réévaluer avec seuil optimal
y_pred_optimal = (y_proba >= best_threshold).astype(int)
print(classification_report(y_test, y_pred_optimal))
```

#### Étape 5.5 - Learning Curves
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train,
    cv=5, scoring='recall',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Taille du dataset')
plt.ylabel('Recall')
plt.title('Learning Curves')
plt.legend()
plt.show()
```

#### Étape 5.6 - Calibration des Probabilités
```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Vérifier la calibration
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Probabilité prédite')
plt.ylabel('Probabilité réelle')
plt.title('Calibration Curve')
plt.show()

# Calibrer si nécessaire (Platt Scaling)
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)
y_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
```

#### Étape 5.7 - Validation des Objectifs Business
```python
# Objectif: Recall ≥ 75%
recall_achieved = recall_score(y_test, y_pred)
objectif_recall = 0.75

if recall_achieved >= objectif_recall:
    print(f"✓ Objectif atteint: Recall = {recall_achieved:.2%} ≥ {objectif_recall:.0%}")
else:
    print(f"✗ Objectif non atteint: Recall = {recall_achieved:.2%} < {objectif_recall:.0%}")
    print("Actions: Ajuster le seuil, essayer SMOTE, optimiser davantage")
```

---

## 6. DÉPLOIEMENT (Deployment)

### Phase de Production

#### Étape 6.1 - Sérialisation du Modèle
```python
import joblib

# Sauvegarder le modèle
joblib.dump(best_model, 'models/best_model.pkl')

# Sauvegarder le preprocessor
joblib.dump(preprocessor, 'models/preprocessor.pkl')

# Sauvegarder les métadonnées
metadata = {
    'model_name': 'RandomForest_balanced',
    'train_date': '2026-03-15',
    'features': list(X_train.columns),
    'metrics': metrics,
    'threshold': best_threshold
}
joblib.dump(metadata, 'models/metadata.pkl')
```

#### Étape 6.2 - Création de l'API

Voir `api/app.py` pour l'implémentation complète.

#### Étape 6.3 - Tests Unitaires
```python
# tests/test_api.py
import pytest
from api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_predict_single(client):
    data = {
        'tenure': 12,
        'MonthlyCharges': 70.0,
        'TotalCharges': 840.0,
        'Contract': 'Month-to-month',
        # ... autres features
    }
    response = client.post('/predict/single', json=data)
    assert response.status_code == 200
    assert 'churn_probability' in response.json
```

#### Étape 6.4 - Monitoring

##### Tracking MLflow
```python
import mlflow

mlflow.set_experiment('churn_prediction')

with mlflow.start_run():
    mlflow.log_params(best_model.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(best_model, 'model')
```

##### Détection de Drift
```python
def calculate_psi(expected, actual, bins=10):
    """
    Population Stability Index (PSI)
    PSI < 0.1: Pas de drift
    0.1 ≤ PSI < 0.2: Drift modéré
    PSI ≥ 0.2: Drift significatif
    """
    expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)
    
    psi = np.sum((actual_percents - expected_percents) * 
                 np.log((actual_percents + 1e-10) / (expected_percents + 1e-10)))
    
    return psi

# Utilisation
psi = calculate_psi(X_train['tenure'], X_new['tenure'])
if psi >= 0.2:
    print("⚠️  Drift détecté → Ré-entraînement nécessaire")
```

---

## 7. EXPLICABILITÉ

### SHAP (SHapley Additive exPlanations)

#### Étape 7.1 - Initialisation
```python
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
```

#### Étape 7.2 - Visualisations

##### Summary Plot (Importance Globale)
```python
shap.summary_plot(shap_values[1], X_test, plot_type='bar')
shap.summary_plot(shap_values[1], X_test)
```

##### Force Plot (Explication Individuelle)
```python
i = 0  # Premier client du test set
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][i],
    X_test.iloc[i],
    matplotlib=True
)
```

##### Dependence Plot
```python
shap.dependence_plot('tenure', shap_values[1], X_test)
```

##### Waterfall Plot
```python
shap.waterfall_plot(shap.Explanation(
    values=shap_values[1][i],
    base_values=explainer.expected_value[1],
    data=X_test.iloc[i],
    feature_names=X_test.columns
))
```

---

## CHECKLIST FINALE

### Avant de livrer, vérifiez:

- [ ] Dataset téléchargé et exploré
- [ ] Au moins 10 nouvelles features créées
- [ ] Pipeline de preprocessing fonctionnel
- [ ] Au moins 5 modèles comparés avec métriques complètes
- [ ] Stratégie de gestion du déséquilibre implémentée
- [ ] Meilleur modèle optimisé (GridSearch ou RandomizedSearch)
- [ ] Recall ≥ 75% sur test set
- [ ] SHAP implémenté avec au moins 3 visualisations
- [ ] API REST fonctionnelle avec tous les endpoints
- [ ] Tests unitaires de l'API écrits et passants
- [ ] Dockerfile créé et testé
- [ ] docker-compose.yml fonctionnel
- [ ] Monitoring MLflow configuré
- [ ] Détection de drift implémentée
- [ ] Rapport de synthèse rédigé (5-10 pages)
- [ ] Présentation préparée (10 slides)
- [ ] Code commenté et documenté
- [ ] README.md à jour

---

**Bonne chance pour votre projet !**
