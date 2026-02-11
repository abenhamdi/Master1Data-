# Dockerfile simplifié pour API de détection de pneumonie
# Version CPU (plus simple pour débuter, pas besoin de nvidia-docker)
# Pour GPU: remplacer par tensorflow/tensorflow:latest-gpu

FROM tensorflow/tensorflow:latest

# Métadonnées
LABEL maintainer="Étudiant Master 1 Data"
LABEL description="Pneumonia Detection API - Version CPU"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Mise à jour et installation dépendances
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copier requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY api/ ./api/
COPY models/ ./models/

# Créer utilisateur non-root
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Exposer le port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Lancement de l'API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
