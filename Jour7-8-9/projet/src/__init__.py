"""
Module src pour le projet Churn Prediction.
Contient les utilitaires de preprocessing et prédiction.
"""

from .preprocessing import (
    ChurnFeatureEngineer,
    load_and_clean_data,
    create_preprocessing_pipeline,
    prepare_data_for_modeling,
    calculate_psi
)

from .predict import (
    ChurnPredictor,
    predict_churn_from_file,
    create_sample_input
)

__all__ = [
    'ChurnFeatureEngineer',
    'load_and_clean_data',
    'create_preprocessing_pipeline',
    'prepare_data_for_modeling',
    'calculate_psi',
    'ChurnPredictor',
    'predict_churn_from_file',
    'create_sample_input'
]

__version__ = '1.0.0'
