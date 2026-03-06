# AIDE-MÉMOIRE - SNIPPETS DE CODE

Référence rapide pour le projet Churn Prediction.

---

## CHARGEMENT ET EXPLORATION

### Imports Essentiels
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             accuracy_score, precision_score, recall_score, f1_score)

import warnings
warnings.filterwarnings('ignore')
```

### Charger le Dataset
```python
df = pd.read_csv('data/telecom_churn.csv')
print(f"Shape: {df.shape}")
df.head()
```

### Exploration Rapide
```python
# Infos générales
df.info()
df.describe()

# Distribution target
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True))

# Valeurs manquantes
print(df.isnull().sum())

# Corrélations (après encodage numérique)
df.corr()['Churn_binary'].sort_values(ascending=False)
```

---

## FEATURE ENGINEERING

### Gestion des Valeurs Manquantes
```python
# TotalCharges manquantes
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)
```

### Features de Ratio
```python
# Dépense moyenne mensuelle
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)

# Ratio charges / tenure
df['ChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

# Ratio TotalCharges / MonthlyCharges
df['ChargeRatio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)
```

### Features d'Agrégation
```python
# Nombre total de services
service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                'StreamingTV', 'StreamingMovies']

df['TotalServices'] = (df[service_cols].apply(lambda x: x.str.contains('Yes|DSL|Fiber'), axis=0)).sum(axis=1)

# Score d'engagement
df['EngagementScore'] = df['tenure'] * df['TotalServices']
```

### Features d'Interaction
```python
# Interaction catégorielle
df['Contract_Internet'] = df['Contract'] + '_' + df['InternetService']

# Interaction numérique
df['Tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
```

### Features Binaires
```python
# Client nouveau (<6 mois)
df['IsNewCustomer'] = (df['tenure'] < 6).astype(int)

# Client premium (>80$ par mois)
df['IsPremium'] = (df['MonthlyCharges'] > 80).astype(int)

# Senior avec internet fiber
df['SeniorFiber'] = ((df['SeniorCitizen'] == 1) & 
                     (df['InternetService'] == 'Fiber optic')).astype(int)
```

### Transformations Mathématiques
```python
# Log transformations
df['LogMonthlyCharges'] = np.log1p(df['MonthlyCharges'])
df['LogTotalCharges'] = np.log1p(df['TotalCharges'])

# Polynomiales
df['Tenure_Squared'] = df['tenure'] ** 2
df['Tenure_Cubed'] = df['tenure'] ** 3
```

---

## ENCODAGE

### Target Encoding
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Churn_binary'] = le.fit_transform(df['Churn'])  # Yes=1, No=0
```

### One-Hot Encoding
```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
encoded = ohe.fit_transform(df[['gender', 'Partner', 'Dependents']])

# Créer DataFrame avec noms de colonnes
encoded_df = pd.DataFrame(
    encoded, 
    columns=ohe.get_feature_names_out(['gender', 'Partner', 'Dependents'])
)
df = pd.concat([df, encoded_df], axis=1)
```

### Pandas get_dummies
```python
df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents'], drop_first=True)
```

### Ordinal Encoding
```python
from sklearn.preprocessing import OrdinalEncoder

contract_order = [['Month-to-month', 'One year', 'Two year']]
oe = OrdinalEncoder(categories=contract_order)
df['Contract_encoded'] = oe.fit_transform(df[['Contract']])
```

### Target Encoding
```python
# Moyenne de la target par modalité
target_mean = df.groupby('PaymentMethod')['Churn_binary'].mean()
df['PaymentMethod_target_enc'] = df['PaymentMethod'].map(target_mean)
```

---

## PREPROCESSING PIPELINE

### Pipeline Scikit-learn Complet
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Séparer les features par type
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['gender', 'Partner', 'Dependents', 'Contract']

# Définir les transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Garder les autres colonnes
)

# Pipeline avec modèle
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Utilisation
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

---

## TRAIN/TEST SPLIT

### Split Stratifié
```python
from sklearn.model_selection import train_test_split

X = df.drop(['Churn', 'Churn_binary', 'customerID'], axis=1)
y = df['Churn_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.2%}")
print(f"Test churn rate: {y_test.mean():.2%}")
```

---

## MODÉLISATION

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]
```

### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42)
dt.fit(X_train, y_train)
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
```

### XGBoost
```python
from xgboost import XGBClassifier

# Calculer scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)
```

### LightGBM
```python
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    is_unbalance=True,  # Gestion déséquilibre
    random_state=42
)
lgbm.fit(X_train, y_train)
```

### SVM
```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, 
          class_weight='balanced', random_state=42)
svm.fit(X_train, y_train)
```

### KNN
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
knn.fit(X_train, y_train)
```

---

## GESTION DU DÉSÉQUILIBRE

### SMOTE
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Avant SMOTE: {y_train.value_counts()}")
print(f"Après SMOTE: {y_train_smote.value_counts()}")
```

### RandomUnderSampler
```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
```

### SMOTE + Tomek Links
```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)
```

---

## CROSS-VALIDATION

### StratifiedKFold
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    rf, X_train, y_train,
    cv=skf,
    scoring='recall',
    n_jobs=-1
)

print(f"CV Recall: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Scores: {cv_scores}")
```

### Multiples Métriques
```python
from sklearn.model_selection import cross_validate

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results = cross_validate(
    rf, X_train, y_train,
    cv=5,
    scoring=scoring,
    n_jobs=-1
)

for metric in scoring:
    score = cv_results[f'test_{metric}']
    print(f"{metric}: {score.mean():.4f} ± {score.std():.4f}")
```

---

## OPTIMISATION HYPERPARAMÈTRES

### GridSearchCV
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"Meilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score CV: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
```

### RandomizedSearchCV
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
    n_iter=100,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
```

---

## ÉVALUATION

### Métriques Complètes
```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Classification report
print(classification_report(y_test, y_pred))

# Métriques individuelles
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

### Matrice de Confusion
```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.xlabel('Prédiction')
plt.ylabel('Réalité')
plt.title('Matrice de Confusion')
plt.show()

# Extraire les valeurs
TN, FP, FN, TP = cm.ravel()
print(f"True Negatives: {TN}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"True Positives: {TP}")
```

### Courbe ROC
```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
```

### Courbe Precision-Recall
```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
```

---

## EXPLICABILITÉ

### Feature Importance (MDI)
```python
# Pour Random Forest / XGBoost
importances = model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1]

# Top 10
plt.figure(figsize=(10, 6))
plt.bar(range(10), importances[indices[:10]])
plt.xticks(range(10), feature_names[indices[:10]], rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 10 Feature Importances (MDI)')
plt.tight_layout()
plt.show()
```

### Permutation Importance
```python
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

sorted_idx = perm_importance.importances_mean.argsort()[::-1][:10]

plt.figure(figsize=(10, 6))
plt.barh(range(10), perm_importance.importances_mean[sorted_idx])
plt.yticks(range(10), X_test.columns[sorted_idx])
plt.xlabel('Permutation Importance')
plt.title('Top 10 Permutation Importances')
plt.tight_layout()
plt.show()
```

### SHAP Values
```python
import shap

# Créer l'explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot (bar)
shap.summary_plot(shap_values[1], X_test, plot_type='bar', max_display=10)

# Summary plot (beeswarm)
shap.summary_plot(shap_values[1], X_test, max_display=10)

# Force plot (1 prédiction)
i = 0
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][i],
    X_test.iloc[i],
    matplotlib=True
)

# Waterfall plot
shap.waterfall_plot(shap.Explanation(
    values=shap_values[1][i],
    base_values=explainer.expected_value[1],
    data=X_test.iloc[i],
    feature_names=X_test.columns
))

# Dependence plot
shap.dependence_plot('tenure', shap_values[1], X_test)
```

---

## SERIALISATION

### Sauvegarder le Modèle
```python
import joblib

# Sauvegarder
joblib.dump(model, 'models/best_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')

# Charger
loaded_model = joblib.load('models/best_model.pkl')
loaded_preprocessor = joblib.load('models/preprocessor.pkl')
```

---

## API FLASK

### app.py Minimal
```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle
model = joblib.load('models/best_model.pkl')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    
    # Preprocessing
    # ... (appliquer le même preprocessing que lors de l'entraînement)
    
    proba = model.predict_proba(df)[0, 1]
    prediction = 'Yes' if proba >= 0.5 else 'No'
    
    return jsonify({
        'churn_probability': float(proba),
        'churn_prediction': prediction,
        'risk_level': 'High' if proba >= 0.7 else 'Medium' if proba >= 0.4 else 'Low'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## MLFLOW

### Tracking d'Expérience
```python
import mlflow

mlflow.set_experiment('churn_prediction')

with mlflow.start_run(run_name='RandomForest_v1'):
    # Log params
    mlflow.log_params(model.get_params())
    
    # Log metrics
    mlflow.log_metric('recall', recall_score(y_test, y_pred))
    mlflow.log_metric('precision', precision_score(y_test, y_pred))
    mlflow.log_metric('roc_auc', roc_auc_score(y_test, y_proba))
    
    # Log model
    mlflow.sklearn.log_model(model, 'model')
    
    # Log artifacts
    mlflow.log_artifact('models/best_model.pkl')
```

---

## DOCKER

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "api/app.py"]
```

### Build et Run
```bash
docker build -t churn-api:latest .
docker run -p 5000:5000 churn-api:latest
```

---

## MONITORING - DRIFT DETECTION

### Population Stability Index (PSI)
```python
def calculate_psi(expected, actual, bins=10):
    expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)
    
    psi = np.sum((actual_percents - expected_percents) * 
                 np.log((actual_percents + 1e-10) / (expected_percents + 1e-10)))
    
    return psi

# Utilisation
psi = calculate_psi(X_train['tenure'], X_new['tenure'])
print(f"PSI: {psi:.4f}")

if psi < 0.1:
    print("Pas de drift")
elif psi < 0.2:
    print("Drift modéré")
else:
    print("Drift significatif → Ré-entraînement nécessaire")
```

---

**Fin de l'aide-mémoire. Bonne chance !**
