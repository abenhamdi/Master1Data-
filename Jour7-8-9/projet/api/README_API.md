# API REST - CHURN PREDICTION

API Flask pour prédire le risque de churn des clients.

## Démarrage

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Lancer l'API
```bash
python api/app.py
```

L'API sera disponible sur `http://localhost:5000`

### Lancer avec Docker
```bash
docker build -t churn-api:latest .
docker run -p 5000:5000 churn-api:latest
```

---

## Endpoints

### 1. Health Check
**GET** `/health`

Vérifier l'état de santé de l'API.

**Réponse**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600.5,
  "version": "1.0.0",
  "timestamp": "2026-03-15T10:30:00"
}
```

**Exemple curl**:
```bash
curl http://localhost:5000/health
```

---

### 2. Prédiction Individuelle
**POST** `/predict/single`

Prédire le risque de churn pour un seul client.

**Body (JSON)**:
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.0,
  "TotalCharges": 1020.0
}
```

**Réponse**:
```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.78,
  "risk_level": "High",
  "threshold_used": 0.5,
  "latency_ms": 15.3,
  "timestamp": "2026-03-15T10:30:00"
}
```

**Exemple curl**:
```bash
curl -X POST http://localhost:5000/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 1020.0
  }'
```

**Exemple Python**:
```python
import requests

url = "http://localhost:5000/predict/single"
data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 85.0,
    "TotalCharges": 1020.0,
    # ... autres features
}

response = requests.post(url, json=data)
print(response.json())
```

---

### 3. Prédiction Batch
**POST** `/predict/batch`

Prédire le risque de churn pour plusieurs clients en une seule requête.

**Body (JSON)**:
```json
[
  {
    "gender": "Male",
    "tenure": 12,
    "MonthlyCharges": 85.0,
    ...
  },
  {
    "gender": "Female",
    "tenure": 24,
    "MonthlyCharges": 70.0,
    ...
  }
]
```

**Réponse**:
```json
{
  "predictions": [
    {
      "index": 0,
      "churn_prediction": "Yes",
      "churn_probability": 0.78,
      "risk_level": "High",
      "threshold_used": 0.5
    },
    {
      "index": 1,
      "churn_prediction": "No",
      "churn_probability": 0.32,
      "risk_level": "Low",
      "threshold_used": 0.5
    }
  ],
  "count": 2,
  "latency_ms": 25.6,
  "timestamp": "2026-03-15T10:30:00"
}
```

**Exemple curl**:
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"gender": "Male", "tenure": 12, "MonthlyCharges": 85.0, ...},
    {"gender": "Female", "tenure": 24, "MonthlyCharges": 70.0, ...}
  ]'
```

---

### 4. Informations sur le Modèle
**GET** `/model/info`

Récupérer les informations et métriques du modèle chargé.

**Réponse**:
```json
{
  "model_loaded": true,
  "model_type": "RandomForestClassifier",
  "model_name": "RandomForest_balanced",
  "recall": 0.82,
  "roc_auc": 0.87,
  "num_features": 35,
  "threshold": 0.5,
  "api_version": "1.0.0",
  "model_path": "/app/models/best_model.pkl"
}
```

**Exemple curl**:
```bash
curl http://localhost:5000/model/info
```

---

### 5. Métriques de Monitoring
**GET** `/metrics`

Récupérer les statistiques d'utilisation de l'API.

**Réponse**:
```json
{
  "total_predictions": 1250,
  "total_errors": 12,
  "uptime_seconds": 7200.5,
  "predictions_per_risk_level": {
    "Low": 400,
    "Medium": 350,
    "High": 300,
    "Critical": 200
  },
  "error_rate": 0.0096,
  "start_time": "2026-03-15T08:00:00",
  "current_time": "2026-03-15T10:00:00"
}
```

**Exemple curl**:
```bash
curl http://localhost:5000/metrics
```

---

### 6. Modifier le Seuil de Décision
**POST** `/threshold`

Changer le seuil de probabilité pour la classification.

**Body (JSON)**:
```json
{
  "threshold": 0.6
}
```

**Réponse**:
```json
{
  "message": "Seuil mis à jour avec succès",
  "new_threshold": 0.6
}
```

**Exemple curl**:
```bash
curl -X POST http://localhost:5000/threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.6}'
```

---

## Niveaux de Risque

| Probabilité | Niveau de Risque |
|-------------|------------------|
| ≥ 0.8       | Critical         |
| 0.6 - 0.8   | High             |
| 0.4 - 0.6   | Medium           |
| < 0.4       | Low              |

---

## Codes de Statut HTTP

| Code | Description |
|------|-------------|
| 200  | Succès |
| 400  | Requête invalide (données manquantes ou incorrectes) |
| 404  | Endpoint non trouvé |
| 500  | Erreur interne du serveur |
| 503  | Service indisponible (modèle non chargé) |

---

## Tests Unitaires

Lancer les tests:
```bash
pytest tests/test_api.py -v
```

---

## Monitoring avec Prometheus

L'API expose des métriques compatibles Prometheus via `/metrics`.

Métriques disponibles:
- `predictions_total`: Nombre total de prédictions
- `errors_total`: Nombre total d'erreurs
- `prediction_latency_seconds`: Latence des prédictions
- `model_accuracy`: Précision actuelle du modèle

---

## Exemples d'Utilisation

### Python avec requests
```python
import requests

# Prédiction simple
url = "http://localhost:5000/predict/single"
data = {
    "gender": "Male",
    "tenure": 12,
    "MonthlyCharges": 85.0,
    # ... autres features
}

response = requests.post(url, json=data)
result = response.json()

print(f"Prédiction: {result['churn_prediction']}")
print(f"Probabilité: {result['churn_probability']:.2%}")
print(f"Niveau de risque: {result['risk_level']}")
```

### JavaScript/Node.js avec fetch
```javascript
const url = 'http://localhost:5000/predict/single';
const data = {
  gender: 'Male',
  tenure: 12,
  MonthlyCharges: 85.0,
  // ... autres features
};

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data),
})
  .then(response => response.json())
  .then(result => {
    console.log('Prédiction:', result.churn_prediction);
    console.log('Probabilité:', result.churn_probability);
  });
```

---

## Sécurité

**TODO pour production**:
- Ajouter une authentification (JWT, API Key)
- Implémenter rate limiting
- Activer HTTPS
- Valider et sanitizer tous les inputs
- Logger toutes les requêtes

---

## Performances

- **Latence cible**: <200ms par prédiction
- **Throughput**: ~1000 prédictions/seconde (selon le matériel)
- **Optimisations**:
  - Batch predictions pour traiter plusieurs clients
  - Caching des prédictions fréquentes
  - Load balancing avec plusieurs instances

---

## Troubleshooting

### Erreur "Modèle non chargé"
Vérifiez que le fichier `models/best_model.pkl` existe.

### Erreur "Features manquantes"
Assurez-vous de fournir toutes les features attendues par le modèle.

### Latence élevée
Utilisez l'endpoint `/predict/batch` pour traiter plusieurs clients à la fois.

---

## Contact

Pour toute question ou problème, contactez l'équipe data science.
