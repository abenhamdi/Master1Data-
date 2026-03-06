"""
API Flask pour la prédiction de churn.
Endpoints: /health, /predict/single, /predict/batch, /model/info
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime
import time

# Ajouter le dossier parent au path pour importer src
sys.path.append(str(Path(__file__).parent.parent))

from src.predict import ChurnPredictor

# Configuration
app = Flask(__name__)
CORS(app)  # Autoriser les requêtes cross-origin

# Chemins vers les modèles
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'best_model.pkl'
METADATA_PATH = Path(__file__).parent.parent / 'models' / 'metadata.pkl'

# Charger le modèle au démarrage
predictor = ChurnPredictor()
model_loaded = False

try:
    if MODEL_PATH.exists():
        predictor.load_model(str(MODEL_PATH), str(METADATA_PATH) if METADATA_PATH.exists() else None)
        model_loaded = True
        print("✓ Modèle chargé avec succès!")
    else:
        print(f"⚠️  Modèle non trouvé: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")

# Métriques de monitoring
metrics = {
    'total_predictions': 0,
    'total_errors': 0,
    'start_time': datetime.now(),
    'predictions_per_risk_level': {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}
}


@app.route('/')
def home():
    """Page d'accueil avec documentation de l'API."""
    return jsonify({
        'message': 'Churn Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict/single': 'POST - Prédiction individuelle',
            '/predict/batch': 'POST - Prédiction batch',
            '/model/info': 'GET - Informations sur le modèle',
            '/metrics': 'GET - Métriques de monitoring'
        },
        'documentation': 'Voir README_API.md pour plus de détails'
    })


@app.route('/health', methods=['GET'])
def health():
    """
    Health check de l'API.
    
    Returns:
        JSON: Statut de santé de l'API
    """
    uptime = (datetime.now() - metrics['start_time']).total_seconds()
    
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'uptime_seconds': uptime,
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict/single', methods=['POST'])
def predict_single():
    """
    Prédiction pour un seul client.
    
    Expected JSON:
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.0,
        ...
    }
    
    Returns:
        JSON: Résultat de la prédiction
    """
    if not model_loaded:
        metrics['total_errors'] += 1
        return jsonify({'error': 'Modèle non chargé'}), 503
    
    try:
        start_time = time.time()
        
        # Récupérer les données
        data = request.get_json()
        
        if not data:
            metrics['total_errors'] += 1
            return jsonify({'error': 'Aucune donnée fournie'}), 400
        
        # Faire la prédiction
        result = predictor.predict(data)
        
        if result is None:
            metrics['total_errors'] += 1
            return jsonify({'error': 'Erreur lors de la prédiction'}), 500
        
        # Ajouter des métadonnées
        latency = time.time() - start_time
        result['latency_ms'] = round(latency * 1000, 2)
        result['timestamp'] = datetime.now().isoformat()
        
        # Mettre à jour les métriques
        metrics['total_predictions'] += 1
        risk_level = result['risk_level']
        metrics['predictions_per_risk_level'][risk_level] += 1
        
        return jsonify(result), 200
    
    except Exception as e:
        metrics['total_errors'] += 1
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Prédiction pour plusieurs clients.
    
    Expected JSON:
    [
        {"gender": "Male", "tenure": 12, ...},
        {"gender": "Female", "tenure": 24, ...}
    ]
    
    Returns:
        JSON: Liste de résultats
    """
    if not model_loaded:
        metrics['total_errors'] += 1
        return jsonify({'error': 'Modèle non chargé'}), 503
    
    try:
        start_time = time.time()
        
        # Récupérer les données
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            metrics['total_errors'] += 1
            return jsonify({'error': 'Données invalides. Attendu: liste de dictionnaires'}), 400
        
        # Convertir en DataFrame
        df = pd.DataFrame(data)
        
        # Faire les prédictions
        results = predictor.predict_batch(df)
        
        # Ajouter des métadonnées
        latency = time.time() - start_time
        
        response = {
            'predictions': results,
            'count': len(results),
            'latency_ms': round(latency * 1000, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        # Mettre à jour les métriques
        metrics['total_predictions'] += len(results)
        for result in results:
            risk_level = result['risk_level']
            metrics['predictions_per_risk_level'][risk_level] += 1
        
        return jsonify(response), 200
    
    except Exception as e:
        metrics['total_errors'] += 1
        return jsonify({'error': str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Informations sur le modèle chargé.
    
    Returns:
        JSON: Métadonnées du modèle
    """
    if not model_loaded:
        return jsonify({'error': 'Modèle non chargé'}), 503
    
    try:
        info = predictor.get_model_info()
        info['api_version'] = '1.0.0'
        info['model_path'] = str(MODEL_PATH)
        
        return jsonify(info), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Métriques de monitoring de l'API.
    
    Returns:
        JSON: Statistiques d'utilisation
    """
    uptime = (datetime.now() - metrics['start_time']).total_seconds()
    
    response = {
        'total_predictions': metrics['total_predictions'],
        'total_errors': metrics['total_errors'],
        'uptime_seconds': uptime,
        'predictions_per_risk_level': metrics['predictions_per_risk_level'],
        'error_rate': metrics['total_errors'] / max(metrics['total_predictions'], 1),
        'start_time': metrics['start_time'].isoformat(),
        'current_time': datetime.now().isoformat()
    }
    
    return jsonify(response), 200


@app.route('/threshold', methods=['POST'])
def set_threshold():
    """
    Modifier le seuil de décision du modèle.
    
    Expected JSON:
    {
        "threshold": 0.6
    }
    
    Returns:
        JSON: Confirmation
    """
    if not model_loaded:
        return jsonify({'error': 'Modèle non chargé'}), 503
    
    try:
        data = request.get_json()
        threshold = data.get('threshold')
        
        if threshold is None or not (0 <= threshold <= 1):
            return jsonify({'error': 'Seuil invalide. Doit être entre 0 et 1'}), 400
        
        predictor.set_threshold(threshold)
        
        return jsonify({
            'message': 'Seuil mis à jour avec succès',
            'new_threshold': threshold
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handler pour les routes non trouvées."""
    return jsonify({'error': 'Endpoint non trouvé'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handler pour les erreurs internes."""
    metrics['total_errors'] += 1
    return jsonify({'error': 'Erreur interne du serveur'}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("CHURN PREDICTION API")
    print("="*60)
    print(f"Modèle chargé: {'✓ Oui' if model_loaded else '✗ Non'}")
    print(f"URL: http://localhost:5000")
    print("="*60 + "\n")
    
    # Lancer l'application
    app.run(host='0.0.0.0', port=5000, debug=False)
