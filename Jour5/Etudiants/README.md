# TP5 - Techniques Avancées & MLOps

**Master 1 Data Engineering - YNOV Montpellier**  
**Enseignant** : BENHAMDI Ayoub  
**Durée** : 4 heures

---

## 📋 Description

Ce TP vous permet de mettre en pratique les techniques avancées de Machine Learning et les concepts MLOps vus en cours :
- Algorithmes sophistiqués (SVM, KNN, XGBoost, LightGBM)
- Optimisation des hyperparamètres (RandomizedSearchCV)
- Réduction de dimensionnalité (PCA)
- MLOps (MLflow, pipelines, monitoring, versioning)

**Cas d'usage** : Détection de fraude bancaire sur un dataset réel de Kaggle.

---

## 🚀 Installation & Setup

### 1. Prérequis

- Python 3.8+ installé
- pip à jour
- (Optionnel) Environnement virtuel recommandé

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Télécharger le dataset

**Option A : Via Kaggle API (recommandé)**

```bash
# 1. Créer un compte Kaggle (gratuit) sur https://www.kaggle.com
# 2. Aller dans Account > API > Create New API Token
#    Cela télécharge kaggle.json
# 3. Placer kaggle.json dans ~/.kaggle/ (Linux/Mac) ou C:\Users\<user>\.kaggle\ (Windows)
# 4. Télécharger le dataset

kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

**Option B : Téléchargement manuel**

1. Aller sur https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Cliquer sur "Download" (nécessite un compte Kaggle)
3. Décompresser `creditcard.csv` dans le dossier `data/`

**Option C : Dataset alternatif plus léger**

```bash
kaggle datasets download -d kartik2112/fraud-detection
```

⚠️ **Important** : Le dataset fait environ 150 MB. Téléchargez-le **avant** le TP !

### 4. Vérifier l'installation

```bash
python -c "import sklearn, xgboost, lightgbm, mlflow; print('✓ Toutes les librairies sont installées')"
```

---

## 📁 Structure du Projet

```
TP5_Etudiants/
│
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── enonce_tp5.md                      # Énoncé détaillé du TP
│
├── tp5_template.py                    # Script Python à compléter
│
├── utils.py                           # Fonctions utilitaires (fournies)
│
└── data/
    ├── README.md                      # Instructions dataset
    └── creditcard.csv                 # Dataset à télécharger (non inclus)
```

---

## 🎯 Comment Commencer ?

### Étape 1 : Lire l'énoncé

Ouvrez et lisez attentivement `enonce_tp5.md` pour comprendre :
- Les objectifs pédagogiques
- La structure du TP (4 parties)
- Les livrables attendus
- Le barème

### Étape 2 : Choisir votre environnement

**Option A : Script Python (recommandé pour débutants)**

```bash
# Ouvrir tp5_template.py dans votre éditeur préféré
code tp5_template.py  # VS Code
# ou
pycharm tp5_template.py  # PyCharm
```

Suivez les commentaires `# TODO:` pour compléter le code.

**Option B : Jupyter Notebook (pour expérimentation)**

```bash
jupyter notebook
# Créer un nouveau notebook et copier/adapter le code du template
```

### Étape 3 : Suivre les 4 parties

1. **Partie 1** (45 min) : Exploration et préparation
2. **Partie 2** (90 min) : Algorithmes avancés
3. **Partie 3** (60 min) : Optimisation & dimensionnalité
4. **Partie 4** (45 min) : Introduction MLOps

---

## 🛠️ Fonctions Utilitaires (utils.py)

Le fichier `utils.py` contient des fonctions pré-écrites pour vous aider :

| Fonction | Usage |
|----------|-------|
| `load_fraud_dataset()` | Charger le dataset Kaggle |
| `plot_confusion_matrix()` | Afficher une matrice de confusion |
| `plot_roc_curve()` | Tracer la courbe ROC |
| `plot_precision_recall_curve()` | Courbe Precision-Recall |
| `compare_models_performance()` | Comparer plusieurs modèles |
| `plot_feature_importance()` | Visualiser l'importance des features |
| `detect_data_drift()` | Détecter un drift entre deux datasets |
| `save_model_info()` | Sauvegarder les métadonnées d'un modèle |

**Exemple d'utilisation** :

```python
from utils import load_fraud_dataset, plot_confusion_matrix

# Charger le dataset
df = load_fraud_dataset('data/creditcard.csv')

# Après avoir entraîné un modèle et fait des prédictions
plot_confusion_matrix(y_test, y_pred, title="Mon Modèle")
```

---

## 💡 Conseils pour Réussir

### Gestion du Temps

- ⏰ **Partie 1** : 45 min - Ne pas passer trop de temps sur les visualisations
- ⏰ **Partie 2** : 90 min - La plus longue, bien gérer les 4 algorithmes
- ⏰ **Partie 3** : 60 min - RandomizedSearchCV peut être long, commencer tôt
- ⏰ **Partie 4** : 45 min - Si retard, prioriser MLflow et pipeline

### Best Practices

✅ **À FAIRE** :
- Fixer `random_state=42` partout pour la reproductibilité
- Utiliser `class_weight='balanced'` ou `scale_pos_weight` (déséquilibre)
- Toujours faire un split **stratifié** (`stratify=y`)
- Privilégier **F1-Score et ROC-AUC** plutôt que l'Accuracy
- Commenter votre code
- Sauvegarder régulièrement (Ctrl+S)

❌ **À ÉVITER** :
- Normaliser les features V1-V28 (déjà normalisées)
- Faire `fit()` ou `fit_transform()` sur le test set (data leakage)
- Se fier uniquement à l'Accuracy
- Copier-coller sans comprendre

### Si le Dataset est Trop Lourd

Pour accélérer les tests durant le développement :

```python
# Charger seulement un échantillon
df = load_fraud_dataset('data/creditcard.csv', sample_size=50000)
```

Entraînez sur l'échantillon, puis relancez sur le dataset complet à la fin.

---

## 📊 Livrables à Rendre

À la fin du TP, vous devez soumettre :

1. **Votre code** :
   - `tp5_votrenom.py` ou `tp5_votrenom.ipynb`
   - Code propre, commenté, exécutable

2. **Modèle sauvegardé** :
   - `fraud_detection_pipeline.pkl` (votre meilleur modèle)

3. **Rapport MLflow** :
   - Screenshots ou export CSV des runs MLflow

4. **README personnel** :
   - Synthèse de vos résultats (meilleur modèle, performances)
   - Difficultés rencontrées
   - Pistes d'amélioration

---

## 🐛 Dépannage

### Erreur : Module not found

```bash
# Vérifier que vous avez installé les dépendances
pip install -r requirements.txt

# Vérifier la version de Python
python --version  # Doit être 3.8+
```

### Erreur : Dataset introuvable

```bash
# Vérifier que le fichier existe
ls data/creditcard.csv

# Si absent, télécharger (voir section Installation)
```

### XGBoost/LightGBM ne s'installe pas

Sur certains systèmes, installer via conda peut résoudre :

```bash
conda install -c conda-forge xgboost lightgbm
```

### MLflow UI ne démarre pas

```bash
# Vérifier qu'aucun mlflow n'est déjà lancé
pkill -f mlflow

# Relancer
mlflow ui --port 5000
```

### Le code est trop lent

- Réduire `n_iter` dans RandomizedSearchCV (ex: 20 au lieu de 50)
- Utiliser `sample_size` pour charger moins de données
- Réduire `cv` de 5 à 3 folds

---

## 📚 Ressources Utiles

### Documentation Officielle

- **Scikit-learn** : https://scikit-learn.org/stable/
- **XGBoost** : https://xgboost.readthedocs.io/
- **LightGBM** : https://lightgbm.readthedocs.io/
- **MLflow** : https://mlflow.org/docs/latest/

### Tutoriels Recommandés

- [Handling Imbalanced Data](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
- [XGBoost Hyperparameter Tuning](https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning)
- [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html)

### Datasets Alternatifs (pour aller plus loin)

- [Credit Card Fraud (alternatif)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)

---

## 🆘 Support

En cas de difficulté pendant le TP :

1. **Consultez l'énoncé** : La réponse est peut-être dans les consignes
2. **Consultez les fonctions utils.py** : Des exemples y sont documentés
3. **Consultez la documentation officielle** : Liens ci-dessus
4. **Levez la main** : Demandez de l'aide à l'enseignant
5. **Collaborez** : Discutez avec vos voisins (sans copier-coller)

---

## 🎓 Objectifs d'Apprentissage

À l'issue de ce TP, vous serez capable de :

- ✅ Gérer un problème de classes déséquilibrées
- ✅ Implémenter et comparer des algorithmes avancés (SVM, KNN, XGBoost, LightGBM)
- ✅ Optimiser des hyperparamètres avec RandomizedSearchCV
- ✅ Appliquer PCA et feature selection
- ✅ Utiliser MLflow pour tracker des expérimentations
- ✅ Créer des pipelines de production robustes
- ✅ Détecter un data drift
- ✅ Assurer la reproductibilité (versioning, random seeds)

---

## 📧 Contact

**Enseignant** : BENHAMDI Ayoub  
**Cours** : Techniques Avancées & MLOps  
**Master 1** : Data Engineering - YNOV Montpellier

---

**Bon courage et bon TP ! 🚀**

*N'oubliez pas : L'objectif n'est pas seulement d'obtenir un bon modèle, mais de comprendre pourquoi il fonctionne et comment le déployer en production.*
