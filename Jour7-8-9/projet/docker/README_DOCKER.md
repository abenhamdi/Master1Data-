# GUIDE DOCKER - CHURN PREDICTION API

Guide complet pour conteneuriser et déployer l'API de prédiction de churn.

---

## PRÉREQUIS

- Docker installé (version 20.10+)
- Docker Compose installé (version 2.0+)

Vérifier les installations:
```bash
docker --version
docker-compose --version
```

---

## BUILD DE L'IMAGE

### Option 1: Build Simple
```bash
docker build -t churn-api:latest .
```

### Option 2: Build avec Tag de Version
```bash
docker build -t churn-api:v1.0.0 .
```

### Option 3: Build avec Arguments
```bash
docker build \
  --build-arg PYTHON_VERSION=3.10 \
  -t churn-api:latest \
  .
```

---

## LANCER L'API

### Option 1: Docker Run Simple
```bash
docker run -p 5000:5000 churn-api:latest
```

### Option 2: Docker Run avec Volumes
```bash
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data:ro \
  churn-api:latest
```

### Option 3: Docker Run en Mode Détaché
```bash
docker run -d \
  --name churn-api \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models:ro \
  --restart unless-stopped \
  churn-api:latest
```

---

## DOCKER COMPOSE

### Lancer tous les services
```bash
docker-compose up -d
```

Services démarrés:
- **churn-api**: API Flask sur port 5000
- **mlflow**: MLflow UI sur port 5001
- **prometheus**: Prometheus sur port 9090 (optionnel)
- **grafana**: Grafana sur port 3000 (optionnel)

### Voir les logs
```bash
# Tous les services
docker-compose logs -f

# Un service spécifique
docker-compose logs -f churn-api
```

### Arrêter les services
```bash
docker-compose down
```

### Arrêter et supprimer les volumes
```bash
docker-compose down -v
```

---

## VÉRIFICATION DE L'API

### Health Check
```bash
curl http://localhost:5000/health
```

### Test de Prédiction
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

---

## ACCÈS AUX SERVICES

| Service | URL | Identifiants |
|---------|-----|--------------|
| API Churn | http://localhost:5000 | - |
| MLflow UI | http://localhost:5001 | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / admin |

---

## COMMANDES UTILES

### Voir les containers en cours
```bash
docker ps
```

### Voir tous les containers (y compris arrêtés)
```bash
docker ps -a
```

### Voir les images
```bash
docker images
```

### Inspecter un container
```bash
docker inspect churn-api
```

### Exécuter une commande dans un container
```bash
docker exec -it churn-api bash
```

### Voir les logs d'un container
```bash
docker logs churn-api
docker logs -f churn-api  # Suivre les logs en temps réel
```

### Arrêter un container
```bash
docker stop churn-api
```

### Démarrer un container arrêté
```bash
docker start churn-api
```

### Supprimer un container
```bash
docker rm churn-api
```

### Supprimer une image
```bash
docker rmi churn-api:latest
```

---

## VOLUMES

### Lister les volumes
```bash
docker volume ls
```

### Inspecter un volume
```bash
docker volume inspect projet_mlflow-data
```

### Supprimer les volumes inutilisés
```bash
docker volume prune
```

---

## MONITORING

### Health Check Docker
Le Dockerfile contient un HEALTHCHECK qui vérifie automatiquement la santé du container:
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:5000/health || exit 1
```

Voir le statut de santé:
```bash
docker inspect --format='{{.State.Health.Status}}' churn-api
```

### Métriques Docker
```bash
# Stats en temps réel
docker stats churn-api

# Stats une fois
docker stats --no-stream churn-api
```

---

## OPTIMISATION DE L'IMAGE

### Voir la taille de l'image
```bash
docker images churn-api:latest
```

### Analyser les layers de l'image
```bash
docker history churn-api:latest
```

### Multi-stage build (TODO pour optimisation)
Pour réduire la taille de l'image, utiliser un multi-stage build:
```dockerfile
# Stage 1: Builder
FROM python:3.10 as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "api/app.py"]
```

---

## DÉPLOIEMENT EN PRODUCTION

### Docker Swarm (Orchestration Simple)
```bash
# Initialiser Swarm
docker swarm init

# Déployer le stack
docker stack deploy -c docker-compose.yml churn-stack

# Lister les services
docker service ls

# Scaler l'API
docker service scale churn-stack_churn-api=3
```

### Kubernetes (Orchestration Avancée)
Créer des manifests Kubernetes:
- Deployment
- Service
- ConfigMap
- Secrets

Exemple de Deployment:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-api
  template:
    metadata:
      labels:
        app: churn-api
    spec:
      containers:
      - name: churn-api
        image: churn-api:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

---

## SÉCURITÉ

### Bonnes Pratiques

1. **Utilisateur non-root**
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```

2. **Scanner les vulnérabilités**
   ```bash
   docker scan churn-api:latest
   ```

3. **Mettre à jour l'image de base régulièrement**
   ```bash
   docker pull python:3.10-slim
   docker build --no-cache -t churn-api:latest .
   ```

4. **Limiter les ressources**
   ```bash
   docker run -d \
     --memory="1g" \
     --cpus="1.0" \
     -p 5000:5000 \
     churn-api:latest
   ```

5. **Utiliser des secrets Docker** (pour les variables sensibles)
   ```bash
   echo "my_secret_value" | docker secret create api_key -
   ```

---

## TROUBLESHOOTING

### Le container ne démarre pas
```bash
# Voir les logs
docker logs churn-api

# Voir les événements
docker events --filter container=churn-api

# Mode interactif pour debug
docker run -it --rm churn-api:latest bash
```

### Le health check échoue
```bash
# Vérifier le health check manuellement
docker exec churn-api curl http://localhost:5000/health

# Voir l'historique du health check
docker inspect churn-api | grep -A 10 Health
```

### Problème de volumes
```bash
# Vérifier les montages
docker inspect churn-api | grep -A 20 Mounts

# Vérifier les permissions
docker exec churn-api ls -la /app/models
```

### Problème de réseau
```bash
# Vérifier les ports exposés
docker port churn-api

# Tester depuis l'hôte
curl http://localhost:5000/health

# Tester depuis le container
docker exec churn-api curl http://localhost:5000/health
```

---

## NETTOYAGE

### Nettoyer les ressources inutilisées
```bash
# Supprimer tous les containers arrêtés
docker container prune

# Supprimer toutes les images non utilisées
docker image prune

# Supprimer tous les volumes non utilisés
docker volume prune

# Nettoyage complet (ATTENTION: supprime tout)
docker system prune -a --volumes
```

---

## REGISTRY

### Pousser l'image vers Docker Hub
```bash
# Tag l'image
docker tag churn-api:latest username/churn-api:latest

# Login
docker login

# Push
docker push username/churn-api:latest
```

### Pousser vers un registry privé
```bash
# Tag
docker tag churn-api:latest registry.example.com/churn-api:latest

# Push
docker push registry.example.com/churn-api:latest
```

---

## RESSOURCES

- [Documentation Docker](https://docs.docker.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Hub](https://hub.docker.com/)

---

**Bonne chance avec le déploiement Docker !**
