# NOTEBOOKS JUPYTER - PROJET CHURN PREDICTION

Ce dossier contient les 3 notebooks du projet fil rouge, un par jour.

---

## NOTEBOOKS DISPONIBLES

### 01_EDA_Feature_Engineering.ipynb (Jour 1 - 7h)

**Objectif**: Explorer le dataset et créer des features avancées

**Sections**:
- 0. Setup et Imports
- 1.1. Exploration Approfondie
  - Analyse univariée
  - Analyse bivariée
  - Insights métier
- 1.2. Feature Engineering Avancé
  - Créer au minimum 10 features
  - Types: Ratio, Agrégation, Interaction, Binaire
- 1.3. Baseline et Validation
  - Logistic Regression
  - Cross-validation

**Livrables**:
- Dataset enrichi
- Module `src/preprocessing.py`

---

### 02_Modeling_Optimization.ipynb (Jour 2 - 7h)

**Objectif**: Comparer des modèles et atteindre Recall ≥ 75%

**Sections**:
- 2.1. Benchmark de Modèles
  - Comparer 5-6 algorithmes
  - Tableau comparatif
  - Courbes ROC et PR
- 2.2. Gestion du Déséquilibre
  - class_weight, SMOTE, undersampling
  - Comparaison des stratégies
- 2.3. Optimisation Avancée
  - GridSearchCV ou RandomizedSearchCV
  - Optimisation des hyperparamètres
- 2.4. Analyse Avancée
  - Learning curves
  - Calibration des probabilités

**Livrables**:
- Meilleur modèle sauvegardé
- Rapport comparatif

---

### 03_Production_MLOps.ipynb (Jour 3 - 7h)

**Objectif**: Déployer le modèle en production avec monitoring

**Sections**:
- 3.1. Explicabilité
  - Feature Importance (MDI)
  - Permutation Importance
  - SHAP Values (4 visualisations)
- 3.2. API REST
  - Création Flask/FastAPI
  - Tests de l'API
- 3.3. Conteneurisation Docker
  - Dockerfile
  - docker-compose.yml
- 3.4. Monitoring et MLOps
  - MLflow tracking
  - Détection de drift (PSI)
  - A/B testing (plan)

**Livrables**:
- API fonctionnelle
- Docker opérationnel
- Rapport de synthèse

---

## COMMENT UTILISER

### 1. Démarrer Jupyter

```bash
cd projet/notebooks
jupyter notebook
```

### 2. Ouvrir le Notebook du Jour

**Jour 1**: `01_EDA_Feature_Engineering.ipynb`

**Jour 2**: `02_Modeling_Optimization.ipynb`

**Jour 3**: `03_Production_MLOps.ipynb`

### 3. Suivre les Instructions

Chaque notebook contient:
- Des cellules de code **à compléter** (marquées `# TODO:`)
- Des explications contextuelles
- Des questions de réflexion

### 4. Exécuter les Cellules

**Shortcut Jupyter**:
- `Shift + Enter`: Exécuter la cellule et passer à la suivante
- `Ctrl + Enter`: Exécuter la cellule
- `A`: Insérer cellule au-dessus
- `B`: Insérer cellule en-dessous

---

## CONSEILS

1. **Suivez l'ordre**: Complétez les notebooks dans l'ordre (Jour 1 → 2 → 3)

2. **Sauvegardez régulièrement**: `Ctrl + S` ou bouton Save

3. **Commentez votre code**: Ajoutez des commentaires pour expliquer votre raisonnement

4. **Testez progressivement**: Exécutez et vérifiez chaque cellule avant de passer à la suivante

5. **Consultez l'aide-mémoire**: `../AIDE_MEMOIRE.md` contient des snippets utiles

6. **Relancez le kernel si nécessaire**: `Kernel > Restart & Run All` pour tester l'ensemble

---

## STRUCTURE DES CELLULES

### Cellule Markdown

Contient les instructions et explications.

**Exemple**:
```markdown
### Mission 1.1.1 - Chargement du Dataset

TODO: Charger le fichier CSV depuis le dossier data/
```

### Cellule Code

À compléter par l'étudiant.

**Exemple**:
```python
# TODO: Charger le dataset
df = ...

print(f"Shape du dataset: {df.shape}")
df.head()
```

---

## CHECKPOINTS

### Jour 1 - 16h00
Présentation rapide (5 min) des features créées.

### Jour 2 - 16h00
Présentation du tableau comparatif de modèles.

### Jour 3 - 16h00
Présentation orale finale (15 min + 5 min Q&A).

---

## DÉPANNAGE

### Le notebook ne s'ouvre pas

Vérifiez que Jupyter est installé:
```bash
pip install jupyter
```

### Erreur "Kernel died"

Redémarrez le kernel: `Kernel > Restart`

### Imports manquants

Installez les dépendances:
```bash
pip install -r ../requirements.txt
```

### Visualisations ne s'affichent pas

Ajoutez en début de notebook:
```python
%matplotlib inline
```

---

## SAUVEGARDE

### Checkpoints Automatiques

Jupyter crée des checkpoints automatiques dans `.ipynb_checkpoints/`

### Export

Exporter le notebook en HTML pour le rendu:
```bash
jupyter nbconvert --to html 01_EDA_Feature_Engineering.ipynb
```

---

## LIENS UTILES

- [Jupyter Documentation](https://jupyter-notebook.readthedocs.io/)
- [Markdown Cheatsheet](https://www.markdownguide.org/cheat-sheet/)
- [Aide-Mémoire Projet](../AIDE_MEMOIRE.md)
- [Guide Méthodologique](../GUIDE_METHODOLOGIQUE.md)

---

**Bon travail sur les notebooks !**
