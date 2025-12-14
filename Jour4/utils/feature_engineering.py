"""
Module de Feature Engineering pour la détection de fraude bancaire
TP4 - Master 1 Data Engineering

Ce module contient les fonctions de création de features avancées.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def create_temporal_features(df):
    """
    Crée des features temporelles à partir de la colonne Time
    
    Args:
        df (pd.DataFrame): DataFrame avec colonne 'Time'
    
    Returns:
        pd.DataFrame: DataFrame avec nouvelles features temporelles
    """
    df = df.copy()
    
    # TODO: Implémenter les features temporelles
    # 1. hour : Heure de la journée (0-23)
    # 2. day : Jour (0 ou 1)
    # 3. hour_sin, hour_cos : Encodage cyclique
    # 4. period : Période de la journée (nuit/matin/après-midi/soir)
    
    # Exemple de structure:
    # df['hour'] = (df['Time'] / 3600) % 24
    # ...
    
    return df


def create_amount_features(df):
    """
    Crée des features sur les montants
    
    Args:
        df (pd.DataFrame): DataFrame avec colonne 'Amount'
    
    Returns:
        pd.DataFrame: DataFrame avec nouvelles features sur montants
    """
    df = df.copy()
    
    # TODO: Implémenter les features sur montants
    # 1. amount_log : Log transformation
    # 2. amount_sqrt : Racine carrée
    # 3. is_zero_amount : Indicateur montant = 0
    # 4. amount_bin : Binning du montant
    
    return df


def create_interaction_features(df, top_features=None):
    """
    Crée des features d'interaction
    
    Args:
        df (pd.DataFrame): DataFrame
        top_features (list): Liste des features à combiner avec Amount
    
    Returns:
        pd.DataFrame: DataFrame avec features d'interaction
    """
    df = df.copy()
    
    # TODO: Implémenter les interactions
    # Multiplier Amount avec les features PCA importantes
    # Exemple: df['V1_amount'] = df['V1'] * df['Amount']
    
    if top_features is None:
        top_features = ['V1', 'V2', 'V3']  # Par défaut
    
    for feature in top_features:
        if feature in df.columns and 'Amount' in df.columns:
            df[f'{feature}_amount'] = df[feature] * df['Amount']
    
    return df


def create_aggregation_features(df, v_features=None):
    """
    Crée des features d'agrégation sur les composantes PCA
    
    Args:
        df (pd.DataFrame): DataFrame
        v_features (list): Liste des features V à agréger
    
    Returns:
        pd.DataFrame: DataFrame avec features agrégées
    """
    df = df.copy()
    
    # TODO: Implémenter les agrégations
    # Somme, moyenne, écart-type des features PCA
    
    if v_features is None:
        v_features = [f'V{i}' for i in range(1, 6)]  # V1-V5 par défaut
    
    available_features = [f for f in v_features if f in df.columns]
    
    if available_features:
        df['v_sum'] = df[available_features].sum(axis=1)
        df['v_mean'] = df[available_features].mean(axis=1)
        df['v_std'] = df[available_features].std(axis=1)
    
    return df


def create_all_features(df, top_corr_features=None):
    """
    Applique toutes les transformations de feature engineering
    
    Args:
        df (pd.DataFrame): DataFrame brut
        top_corr_features (list): Features les plus corrélées avec Class
    
    Returns:
        pd.DataFrame: DataFrame avec toutes les nouvelles features
    """
    df = df.copy()
    
    # Appliquer toutes les transformations
    df = create_temporal_features(df)
    df = create_amount_features(df)
    df = create_interaction_features(df, top_corr_features)
    df = create_aggregation_features(df)
    
    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer Scikit-Learn pour le Feature Engineering
    Compatible avec les Pipelines
    """
    
    def __init__(self, top_corr_features=None):
        """
        Args:
            top_corr_features (list): Features à utiliser pour interactions
        """
        self.top_corr_features = top_corr_features
    
    def fit(self, X, y=None):
        """Fit (pas de paramètres à apprendre)"""
        return self
    
    def transform(self, X):
        """
        Applique les transformations de feature engineering
        
        Args:
            X (pd.DataFrame): Features
        
        Returns:
            pd.DataFrame: Features transformées
        """
        return create_all_features(X, self.top_corr_features)


# Fonctions utilitaires

def get_period(hour):
    """
    Retourne la période de la journée
    
    Args:
        hour (float): Heure (0-23)
    
    Returns:
        str: Période (night/morning/afternoon/evening)
    """
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'evening'


def get_top_correlated_features(df, target_col='Class', n=10):
    """
    Retourne les N features les plus corrélées avec la cible
    
    Args:
        df (pd.DataFrame): DataFrame
        target_col (str): Nom de la colonne cible
        n (int): Nombre de features à retourner
    
    Returns:
        list: Liste des features
    """
    if target_col not in df.columns:
        raise ValueError(f"Colonne {target_col} non trouvée")
    
    # Calculer les corrélations
    correlations = df.corr()[target_col].abs()
    
    # Exclure la cible elle-même
    correlations = correlations.drop(target_col)
    
    # Trier et retourner le top N
    top_features = correlations.nlargest(n).index.tolist()
    
    return top_features


if __name__ == "__main__":
    # Test du module
    print("Module Feature Engineering chargé avec succès !")
    print("\nFonctions disponibles:")
    print("- create_temporal_features()")
    print("- create_amount_features()")
    print("- create_interaction_features()")
    print("- create_aggregation_features()")
    print("- create_all_features()")
    print("- FeatureEngineer (Transformer)")
    print("- get_top_correlated_features()")

