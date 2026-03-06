# Dataset - Telecom Churn Prediction

## Source
Dataset Kaggle: **mnassrib/telecom-churn-prediction**

URL: https://www.kaggle.com/datasets/mnassrib/telecom-churn-prediction

## Description
Dataset de churn client pour une entreprise de télécommunications avec **7043 clients** et **21 features**.

## Variables

### Informations Client
- `customerID`: Identifiant unique
- `gender`: Genre (Male, Female)
- `SeniorCitizen`: Senior (0: Non, 1: Oui)
- `Partner`: A un partenaire (Yes, No)
- `Dependents`: A des personnes à charge (Yes, No)
- `tenure`: Nombre de mois comme client (0-72)

### Services Souscrits
- `PhoneService`: Service téléphonique (Yes, No)
- `MultipleLines`: Lignes multiples (Yes, No, No phone service)
- `InternetService`: Type internet (DSL, Fiber optic, No)
- `OnlineSecurity`: Sécurité en ligne (Yes, No, No internet service)
- `OnlineBackup`: Sauvegarde en ligne (Yes, No, No internet service)
- `DeviceProtection`: Protection appareil (Yes, No, No internet service)
- `TechSupport`: Support technique (Yes, No, No internet service)
- `StreamingTV`: TV en streaming (Yes, No, No internet service)
- `StreamingMovies`: Films en streaming (Yes, No, No internet service)

### Informations Contractuelles
- `Contract`: Type de contrat (Month-to-month, One year, Two year)
- `PaperlessBilling`: Facturation électronique (Yes, No)
- `PaymentMethod`: Méthode de paiement (Electronic check, Mailed check, Bank transfer, Credit card)
- `MonthlyCharges`: Montant mensuel facturé (18.25 - 118.75)
- `TotalCharges`: Montant total facturé (18.8 - 8684.8)

### Target
- `Churn`: Client parti (Yes, No) **← Variable cible**

## Statistiques Clés

```
Nombre d'observations: 7043
Nombre de features: 20 (+ 1 target)
Taux de churn: ~26.5%
Classes déséquilibrées: Oui
```

## Caractéristiques

### Déséquilibre des classes
- No churn: ~73.5% (5174 clients)
- Churn: ~26.5% (1869 clients)
- Ratio: 2.77:1

### Types de variables
- Numériques: 3 (SeniorCitizen, tenure, MonthlyCharges, TotalCharges)
- Catégorielles: 17
- Target: 1 (binaire)

### Valeurs manquantes
- TotalCharges: 11 valeurs manquantes (remplaçables par 0 ou tenure × MonthlyCharges)

## Téléchargement

```bash
# Via script Python
python download_data.py

# Via Kaggle CLI
kaggle datasets download -d mnassrib/telecom-churn-prediction
unzip telecom-churn-prediction.zip
```

## Préparation recommandée

1. Gestion des valeurs manquantes dans `TotalCharges`
2. Encodage des variables catégorielles
3. Feature engineering (ratio, interactions, agrégations)
4. Normalisation des features numériques
5. Stratégie de gestion du déséquilibre (class_weight, SMOTE, etc.)
