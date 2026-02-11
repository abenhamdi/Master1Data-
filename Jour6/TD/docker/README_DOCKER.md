# GUIDE DOCKER - API PNEUMONIA DETECTION

## PRÉREQUIS

### Installation Docker

**Linux** :
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**macOS** : Installer [Docker Desktop](https://www.docker.com/products/docker-desktop)

**Windows** : Installer [Docker Desktop](https://www.docker.com/products/docker-desktop) avec WSL2

### Vérification
```bash
docker --version
docker compose version
```

---

## UTILISATION RAPIDE

### 1. Build de l'image

Depuis le dossier `TD/` :

```bash
cd docker
docker build -t pneumonia-api:latest .
```

### 2. Lancement avec Docker Compose

```bash
docker compose up -d
```

### 3. Vérification

```bash
# Vérifier que le container tourne
docker ps

# Logs en temps réel
docker compose logs -f pneumonia-api

# Tester l'API
curl http://localhost:8000/health
```

### 4. Test de prédiction

```bash
# Prédiction sur une image
curl -X POST -F "file=@../data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg" \
  http://localhost:8000/predict
```

Réponse attendue :
```json
{
  "class": "PNEUMONIA",
  "probability": 0.9234,
  "confidence": "High",
  "processing_time_ms": 156,
  "model_version": "1.0"
}
```

### 5. Arrêt

```bash
docker compose down
```

---

## COMMANDES UTILES

### Gestion des containers

```bash
# Démarrer
docker compose up -d

# Arrêter
docker compose stop

# Redémarrer
docker compose restart

# Supprimer (avec volumes)
docker compose down -v

# Voir les logs
docker compose logs -f

# Entrer dans le container
docker exec -it pneumonia-api bash
```

### Gestion des images

```bash
# Lister les images
docker images

# Supprimer une image
docker rmi pneumonia-api:latest

# Rebuild sans cache
docker compose build --no-cache

# Nettoyer les images non utilisées
docker image prune -a
```

---

## CONFIGURATION GPU (OPTIONNEL)

### Prérequis GPU

1. **Carte NVIDIA** avec drivers installés
2. **NVIDIA Container Toolkit** :

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Vérification GPU

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### Activation GPU

1. **Modifier le Dockerfile** :
```dockerfile
# Ligne 1 du Dockerfile
FROM tensorflow/tensorflow:2.18.0-gpu  # Au lieu de 2.18.0
```

2. **Décommenter dans docker-compose.yml** :
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

3. **Rebuild et relancer** :
```bash
docker compose build
docker compose up -d
```

---

## STRUCTURE DES FICHIERS

```
TD/
├── docker/
│   ├── Dockerfile              # Définition de l'image
│   ├── docker-compose.yml      # Orchestration
│   └── README_DOCKER.md        # Ce fichier
├── api/
│   └── app.py                  # Code de l'API
├── models/
│   └── resnet50_best.h5        # Modèle entraîné
└── requirements.txt            # Dépendances Python
```

---

## DÉPANNAGE

### Problème 1 : Port 8000 déjà utilisé

**Erreur** : `bind: address already in use`

**Solution** :
```bash
# Trouver le processus
lsof -i :8000

# Tuer le processus
kill -9 <PID>

# Ou changer le port dans docker-compose.yml
ports:
  - "8001:8000"  # Utiliser 8001 au lieu de 8000
```

### Problème 2 : Modèle non trouvé

**Erreur** : `FileNotFoundError: models/resnet50_best.h5`

**Solution** :
```bash
# Vérifier que le modèle existe
ls -lh ../models/

# S'assurer que le volume est bien monté
docker inspect pneumonia-api | grep Mounts -A 10
```

### Problème 3 : Out of Memory (OOM)

**Erreur** : Container killed (exit code 137)

**Solution** :
```yaml
# Augmenter la limite dans docker-compose.yml
mem_limit: 4g  # Au lieu de 2g
```

### Problème 4 : Build échoue

**Erreur** : `failed to solve with frontend dockerfile.v0`

**Solution** :
```bash
# Nettoyer le cache Docker
docker builder prune -a

# Rebuild sans cache
docker compose build --no-cache
```

### Problème 5 : GPU non détecté

**Erreur** : `Could not load dynamic library 'libcudart.so.12'`

**Solution** :
```bash
# Vérifier nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Si échec, réinstaller nvidia-container-toolkit
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker
```

---

## OPTIMISATIONS

### 1. Réduire la taille de l'image

```dockerfile
# Utiliser une image plus légère (dans Dockerfile)
FROM python:3.11-slim

# Installer TensorFlow CPU uniquement
RUN pip install tensorflow-cpu==2.18.0
```

### 2. Cache des layers

```dockerfile
# Copier requirements.txt AVANT le code
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copier le code après (changements fréquents)
COPY api/ ./api/
```

### 3. Multi-stage build (avancé)

```dockerfile
# Stage 1: Builder
FROM python:3.11 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY api/ ./api/
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## TESTS

### Test de charge (optionnel)

```bash
# Installer Apache Bench
sudo apt-get install apache2-utils

# Test avec 100 requêtes, 10 concurrentes
ab -n 100 -c 10 http://localhost:8000/health
```

### Test avec Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Prédiction
with open("test_image.jpeg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

---

## SÉCURITÉ

### Bonnes pratiques appliquées

✅ **Utilisateur non-root** : Container tourne avec user `appuser` (UID 1000)
✅ **Volumes read-only** : Modèles montés en `:ro`
✅ **Health checks** : Monitoring automatique
✅ **Limites de ressources** : `mem_limit` et `cpus` définis
✅ **Pas de secrets hardcodés** : Utiliser variables d'environnement
✅ **Image officielle** : Base TensorFlow officielle (vérifiée)

### Recommandations supplémentaires

- Ne jamais committer `.env` avec secrets
- Utiliser Docker secrets pour production
- Scanner l'image avec Trivy :
  ```bash
  docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image pneumonia-api:latest
  ```

---

## RESSOURCES

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [TensorFlow Docker](https://www.tensorflow.org/install/docker)
- [FastAPI Docker](https://fastapi.tiangolo.com/deployment/docker/)

---

**Bon courage pour le déploiement !**
