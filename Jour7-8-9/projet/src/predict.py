"""
Module de prédiction pour le projet Churn Prediction.
Contient des classes et fonctions pour charger le modèle et faire des prédictions.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class ChurnPredictor:
    """
    Classe pour gérer les prédictions de churn.
    """
    
    def __init__(self, model_path=None, metadata_path=None):
        """
        Initialiser le predicteur.
        
        Args:
            model_path (str): Chemin vers le modèle sauvegardé
            metadata_path (str): Chemin vers les métadonnées
        """
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.threshold = 0.5
        
        if model_path:
            self.load_model(model_path, metadata_path)
    
    def load_model(self, model_path, metadata_path=None):
        """
        Charger le modèle et les métadonnées.
        
        Args:
            model_path (str): Chemin vers le modèle
            metadata_path (str): Chemin vers les métadonnées
        """
        try:
            self.model = joblib.load(model_path)
            print(f"Modèle chargé depuis: {model_path}")
            
            if metadata_path:
                self.metadata = joblib.load(metadata_path)
                self.feature_names = self.metadata.get('features', None)
                self.threshold = self.metadata.get('threshold', 0.5)
                print(f"Métadonnées chargées depuis: {metadata_path}")
            
            return True
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            return False
    
    def predict(self, X, return_proba=True):
        """
        Faire une prédiction.
        
        Args:
            X (pd.DataFrame ou dict): Features d'un client
            return_proba (bool): Retourner les probabilités
        
        Returns:
            dict: Résultat de la prédiction
        """
        if self.model is None:
            raise ValueError("Modèle non chargé. Utilisez load_model() d'abord.")
        
        # Convertir dict en DataFrame si nécessaire
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        # Vérifier les features
        if self.feature_names:
            # S'assurer que toutes les features attendues sont présentes
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                print(f"Attention: Features manquantes: {missing_features}")
                # Ajouter les features manquantes avec 0
                for feat in missing_features:
                    X[feat] = 0
            
            # Réordonner les colonnes
            X = X[self.feature_names]
        
        # Prédiction
        try:
            proba = self.model.predict_proba(X)[0, 1]
            prediction = 1 if proba >= self.threshold else 0
            prediction_label = 'Yes' if prediction == 1 else 'No'
            
            result = {
                'churn_prediction': prediction_label,
                'churn_probability': float(proba),
                'risk_level': self._get_risk_level(proba),
                'threshold_used': self.threshold
            }
            
            if not return_proba:
                result.pop('churn_probability')
            
            return result
        
        except Exception as e:
            print(f"Erreur lors de la prédiction: {e}")
            return None
    
    def predict_batch(self, X):
        """
        Faire des prédictions sur un batch de clients.
        
        Args:
            X (pd.DataFrame): Features de plusieurs clients
        
        Returns:
            list: Liste de résultats de prédiction
        """
        results = []
        
        for idx in range(len(X)):
            row = X.iloc[[idx]]
            result = self.predict(row)
            if result:
                result['index'] = idx
                results.append(result)
        
        return results
    
    def _get_risk_level(self, probability):
        """
        Déterminer le niveau de risque selon la probabilité.
        
        Args:
            probability (float): Probabilité de churn
        
        Returns:
            str: Niveau de risque (Low, Medium, High, Critical)
        """
        if probability >= 0.8:
            return 'Critical'
        elif probability >= 0.6:
            return 'High'
        elif probability >= 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def explain_prediction(self, X, top_n=5):
        """
        Expliquer une prédiction (Feature Importance).
        
        Args:
            X (pd.DataFrame ou dict): Features d'un client
            top_n (int): Nombre de top features à retourner
        
        Returns:
            dict: Explication de la prédiction
        """
        if self.model is None:
            raise ValueError("Modèle non chargé.")
        
        # Convertir dict en DataFrame si nécessaire
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        # Prédiction
        result = self.predict(X)
        
        # Feature importance (si disponible)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(importances))]
            
            # Créer DataFrame des importances
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            top_features = importance_df.head(top_n).to_dict('records')
            
            result['top_features'] = top_features
        else:
            result['top_features'] = "Feature importance non disponible pour ce modèle"
        
        return result
    
    def get_model_info(self):
        """
        Récupérer les informations sur le modèle.
        
        Returns:
            dict: Informations du modèle
        """
        info = {
            'model_loaded': self.model is not None,
            'model_type': type(self.model).__name__ if self.model else None,
            'threshold': self.threshold
        }
        
        if self.metadata:
            info.update({
                'model_name': self.metadata.get('model_name'),
                'recall': self.metadata.get('recall'),
                'roc_auc': self.metadata.get('roc_auc'),
                'num_features': len(self.feature_names) if self.feature_names else None
            })
        
        return info
    
    def set_threshold(self, threshold):
        """
        Définir un nouveau seuil de décision.
        
        Args:
            threshold (float): Nouveau seuil (entre 0 et 1)
        """
        if 0 <= threshold <= 1:
            self.threshold = threshold
            print(f"Seuil de décision mis à jour: {threshold}")
        else:
            print("Erreur: Le seuil doit être entre 0 et 1")


def predict_churn_from_file(model_path, data_path, output_path=None):
    """
    Faire des prédictions à partir d'un fichier CSV.
    
    Args:
        model_path (str): Chemin vers le modèle
        data_path (str): Chemin vers le CSV de données
        output_path (str): Chemin pour sauvegarder les résultats (optionnel)
    
    Returns:
        pd.DataFrame: DataFrame avec les prédictions
    """
    # Charger le modèle
    predictor = ChurnPredictor(model_path)
    
    # Charger les données
    df = pd.read_csv(data_path)
    print(f"Données chargées: {df.shape}")
    
    # Prédictions
    results = predictor.predict_batch(df)
    
    # Créer DataFrame de résultats
    results_df = pd.DataFrame(results)
    
    # Fusionner avec les données originales
    output_df = df.copy()
    output_df['churn_prediction'] = results_df['churn_prediction']
    output_df['churn_probability'] = results_df['churn_probability']
    output_df['risk_level'] = results_df['risk_level']
    
    # Sauvegarder si demandé
    if output_path:
        output_df.to_csv(output_path, index=False)
        print(f"Résultats sauvegardés dans: {output_path}")
    
    return output_df


def create_sample_input():
    """
    Créer un exemple d'input pour tester la prédiction.
    
    Returns:
        dict: Exemple de features client
    """
    sample = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.0,
        'TotalCharges': 1020.0
    }
    
    return sample


if __name__ == "__main__":
    print("Module predict chargé avec succès!")
    print("\nClasses/Fonctions disponibles:")
    print("- ChurnPredictor: Classe principale pour les prédictions")
    print("- predict_churn_from_file: Prédire à partir d'un fichier CSV")
    print("- create_sample_input: Créer un exemple d'input")
    
    # Exemple d'utilisation
    print("\n" + "="*60)
    print("EXEMPLE D'UTILISATION")
    print("="*60)
    
    sample_input = create_sample_input()
    print("\nSample input:")
    for key, value in sample_input.items():
        print(f"  {key}: {value}")
    
    print("\n# Pour faire une prédiction:")
    print("predictor = ChurnPredictor('models/best_model.pkl', 'models/metadata.pkl')")
    print("result = predictor.predict(sample_input)")
    print("print(result)")
