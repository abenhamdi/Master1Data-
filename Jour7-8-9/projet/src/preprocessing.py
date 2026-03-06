"""
Module de preprocessing pour le projet Churn Prediction.
Contient des fonctions réutilisables pour le feature engineering et la préparation des données.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer Scikit-learn pour créer des features avancées.
    """
    
    def __init__(self, create_ratios=True, create_aggregations=True, 
                 create_interactions=True, create_binaries=True):
        self.create_ratios = create_ratios
        self.create_aggregations = create_aggregations
        self.create_interactions = create_interactions
        self.create_binaries = create_binaries
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Features de ratio
        if self.create_ratios:
            X_copy = self._create_ratio_features(X_copy)
        
        # Features d'agrégation
        if self.create_aggregations:
            X_copy = self._create_aggregation_features(X_copy)
        
        # Features d'interaction
        if self.create_interactions:
            X_copy = self._create_interaction_features(X_copy)
        
        # Features binaires
        if self.create_binaries:
            X_copy = self._create_binary_features(X_copy)
        
        return X_copy
    
    def _create_ratio_features(self, df):
        """Créer des features de ratio."""
        # Dépense moyenne mensuelle
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Ratio charges / tenure
        df['ChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        
        # Ratio TotalCharges / MonthlyCharges
        df['ChargeRatio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)
        
        return df
    
    def _create_aggregation_features(self, df):
        """Créer des features d'agrégation."""
        # Nombre total de services
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                        'StreamingTV', 'StreamingMovies']
        
        # Compter les services (Yes ou Fiber optic/DSL)
        df['TotalServices'] = 0
        for col in service_cols:
            if col in df.columns:
                if col == 'InternetService':
                    df['TotalServices'] += (df[col].isin(['DSL', 'Fiber optic'])).astype(int)
                else:
                    df['TotalServices'] += (df[col] == 'Yes').astype(int)
        
        # Score d'engagement
        df['EngagementScore'] = df['tenure'] * df['TotalServices']
        
        return df
    
    def _create_interaction_features(self, df):
        """Créer des features d'interaction."""
        # Interaction numérique
        df['Tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
        
        # Interaction catégorielle (si colonnes existent)
        if 'Contract' in df.columns and 'InternetService' in df.columns:
            df['Contract_Internet'] = df['Contract'] + '_' + df['InternetService']
        
        return df
    
    def _create_binary_features(self, df):
        """Créer des features binaires."""
        # Client nouveau (<6 mois)
        df['IsNewCustomer'] = (df['tenure'] < 6).astype(int)
        
        # Client premium (>80$ par mois)
        df['IsPremium'] = (df['MonthlyCharges'] > 80).astype(int)
        
        # Senior avec internet fiber
        if 'SeniorCitizen' in df.columns and 'InternetService' in df.columns:
            df['SeniorFiber'] = ((df['SeniorCitizen'] == 1) & 
                                 (df['InternetService'] == 'Fiber optic')).astype(int)
        
        return df


def load_and_clean_data(filepath):
    """
    Charger et nettoyer le dataset.
    
    Args:
        filepath (str): Chemin vers le fichier CSV
    
    Returns:
        pd.DataFrame: Dataset nettoyé
    """
    df = pd.read_csv(filepath)
    
    # Gestion des valeurs manquantes dans TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)
    
    # Supprimer customerID (non informatif)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    return df


def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Créer un pipeline de preprocessing complet.
    
    Args:
        numerical_features (list): Liste des features numériques
        categorical_features (list): Liste des features catégorielles
    
    Returns:
        sklearn.compose.ColumnTransformer: Pipeline de preprocessing
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def prepare_data_for_modeling(df, target_col='Churn'):
    """
    Préparer les données pour la modélisation.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Nom de la colonne target
    
    Returns:
        tuple: (X, y, feature_names)
    """
    # Encoder la target
    if df[target_col].dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])
    else:
        y = df[target_col].values
    
    # Features
    X = df.drop(target_col, axis=1)
    
    # Encoder les catégorielles avec pd.get_dummies
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    feature_names = X_encoded.columns.tolist()
    
    return X_encoded, y, feature_names


def get_feature_names_after_preprocessing(preprocessor, original_features):
    """
    Récupérer les noms des features après preprocessing.
    
    Args:
        preprocessor: Pipeline de preprocessing
        original_features: Liste des features originales
    
    Returns:
        list: Noms des features après transformation
    """
    try:
        feature_names = []
        
        for name, transformer, features in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                # OneHotEncoder génère de nouveaux noms
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(features))
                else:
                    # Fallback pour anciennes versions
                    feature_names.extend([f"{feat}_{cat}" 
                                         for feat in features 
                                         for cat in transformer.categories_])
            elif name == 'remainder':
                # Colonnes non transformées
                remaining_features = [f for f in original_features 
                                     if f not in feature_names]
                feature_names.extend(remaining_features)
        
        return feature_names
    except Exception as e:
        print(f"Erreur lors de l'extraction des noms de features: {e}")
        return None


# Fonction utilitaire pour calculer le PSI (Population Stability Index)
def calculate_psi(expected, actual, bins=10):
    """
    Calculer le Population Stability Index (PSI) pour détecter le drift.
    
    PSI < 0.1: Pas de drift
    0.1 ≤ PSI < 0.2: Drift modéré
    PSI ≥ 0.2: Drift significatif
    
    Args:
        expected (array-like): Distribution attendue (train)
        actual (array-like): Distribution actuelle (production)
        bins (int): Nombre de bins pour l'histogramme
    
    Returns:
        float: Valeur du PSI
    """
    expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)
    
    # Éviter division par zéro
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi = np.sum((actual_percents - expected_percents) * 
                 np.log(actual_percents / expected_percents))
    
    return psi


if __name__ == "__main__":
    # Exemple d'utilisation
    print("Module preprocessing chargé avec succès!")
    print("\nFonctions disponibles:")
    print("- ChurnFeatureEngineer: Transformer pour feature engineering")
    print("- load_and_clean_data: Charger et nettoyer le dataset")
    print("- create_preprocessing_pipeline: Créer un pipeline de preprocessing")
    print("- prepare_data_for_modeling: Préparer les données pour la modélisation")
    print("- calculate_psi: Calculer le PSI pour détection de drift")
