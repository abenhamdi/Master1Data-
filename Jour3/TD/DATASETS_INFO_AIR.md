# Informations sur le Dataset - Global Air Pollution

## 📊 Global Air Pollution Dataset

### Source principale

**Kaggle - Global Air Pollution Dataset**
- **URL** : https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset
- **Auteur** : Hasib Al Muzdadid
- **Année** : 2022
- **Licence** : Open Data

---

## 📝 Description du Dataset

### Contexte

Ce dataset contient des mesures de qualité de l'air pour plus de 23,000 villes dans le monde, collectées entre 2017 et 2022. Il est basé sur l'Air Quality Index (AQI) de l'EPA (Environmental Protection Agency) américaine.

### Citation

```
Hasib Al Muzdadid (2022). Global Air Pollution Dataset. Kaggle.
https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset
```

### Caractéristiques

- **Nombre d'observations** : 23,000+ villes
- **Période** : 2017-2022
- **Couverture** : Mondiale (tous les continents)
- **Taille** : ~50 MB
- **Format** : CSV

---

## 🔬 Variables

### Variables principales

| Variable | Type | Description |
|----------|------|-------------|
| `Country` | Texte | Pays |
| `City` | Texte | Ville |
| `AQI Value` | Numérique | Indice de Qualité de l'Air (0-500) |
| `AQI Category` | Catégorielle | Catégorie de qualité (Good, Moderate, etc.) |
| `CO AQI Value` | Numérique | Indice pour le monoxyde de carbone |
| `Ozone AQI Value` | Numérique | Indice pour l'ozone (O3) |
| `NO2 AQI Value` | Numérique | Indice pour le dioxyde d'azote |
| `PM2.5 AQI Value` | Numérique | Indice pour les particules fines |
| `lat` | Numérique | Latitude |
| `lng` | Numérique | Longitude |

### Air Quality Index (AQI)

L'AQI est un indice standardisé qui mesure la qualité de l'air :

| Catégorie | AQI | Couleur | Signification |
|-----------|-----|---------|---------------|
| **Good** | 0-50 | Vert | Air de bonne qualité |
| **Moderate** | 51-100 | Jaune | Acceptable, mais risque pour personnes sensibles |
| **Unhealthy for Sensitive Groups** | 101-150 | Orange | Risque pour groupes sensibles |
| **Unhealthy** | 151-200 | Rouge | Risque pour toute la population |
| **Very Unhealthy** | 201-300 | Violet | Alerte sanitaire |
| **Hazardous** | 301+ | Marron | Urgence sanitaire |

---

## 📊 Statistiques descriptives

### Distribution géographique

- **Pays couverts** : ~200 pays
- **Continents** : Tous (Afrique, Amérique, Asie, Europe, Océanie)
- **Villes les plus polluées** : Principalement en Asie du Sud et Moyen-Orient

### Distribution de l'AQI

```
Minimum : 0 (air très pur)
Maximum : 500+ (pollution extrême)
Moyenne : ~80-100 (Moderate)
Médiane : ~70
```

### Répartition des catégories (approximative)

- Good (0-50) : ~30%
- Moderate (51-100) : ~40%
- Unhealthy for Sensitive Groups (101-150) : ~15%
- Unhealthy (151-200) : ~10%
- Very Unhealthy (201-300) : ~4%
- Hazardous (301+) : ~1%

---

## 🎯 Utilisation pédagogique

### Pourquoi ce dataset ?

✅ **Avantages :**
1. **Données très récentes** (2017-2022)
2. **Couverture mondiale** (23,000+ villes)
3. **Taille idéale** (~50 MB - ni trop petit, ni trop gros)
4. **Bien structuré** : Pas de valeurs manquantes complexes
5. **Contexte pertinent** : Enjeu de santé publique majeur
6. **Variables compréhensibles** : Polluants connus (PM2.5, NO2, etc.)
7. **Impact pédagogique** : Sensibilisation environnementale

⚠️ **Limitations :**
1. Données agrégées (pas de séries temporelles détaillées)
2. Certaines villes peuvent manquer de données pour certains polluants
3. Qualité des mesures variable selon les pays

### Objectifs d'apprentissage

Ce dataset permet de travailler :
- ✅ Classification binaire (bon/mauvais air)
- ✅ Classification multi-classes (6 catégories AQI)
- ✅ Régression (prédiction de la valeur AQI)
- ✅ Analyse géographique (visualisation mondiale)
- ✅ Feature importance (quels polluants sont critiques)
- ✅ Comparaison de modèles ML

---

## 🌍 Contexte Environnemental

### Impact sur la Santé

Selon l'Organisation Mondiale de la Santé (OMS) :
- **99%** de la population mondiale respire un air pollué
- **7 millions** de décès prématurés par an dus à la pollution
- **Principal risque** environnemental pour la santé

### Polluants Principaux

1. **PM2.5** (Particules fines <2.5 μm)
   - Pénètrent profondément dans les poumons
   - Causent maladies cardiovasculaires et respiratoires
   - Principal indicateur de pollution

2. **NO2** (Dioxyde d'azote)
   - Provient des véhicules et industries
   - Irrite les voies respiratoires
   - Contribue au smog

3. **O3** (Ozone)
   - Formé par réaction chimique (soleil + polluants)
   - Irrite les poumons
   - Aggrave l'asthme

4. **CO** (Monoxyde de carbone)
   - Provient de combustion incomplète
   - Réduit l'oxygénation du sang
   - Dangereux en espace confiné

---

## 🔄 Alternatives de datasets

Si vous souhaitez varier ou proposer des alternatives :

### 1. UCI Air Quality Dataset
- **URL** : https://archive.ics.uci.edu/ml/datasets/Air+Quality
- **Taille** : Plus petit
- **Période** : 2004-2005 (plus ancien)
- **Localisation** : 1 ville italienne
- **Difficulté** : Similaire

### 2. OpenAQ
- **URL** : https://openaq.org/
- **Taille** : Variable
- **Période** : Temps réel
- **Couverture** : Mondiale
- **Difficulté** : Plus avancé (API)

### 3. WHO Global Air Quality Database
- **URL** : https://www.who.int/data/gho/data/themes/air-pollution
- **Taille** : Moyenne
- **Période** : Mise à jour régulière
- **Couverture** : Mondiale
- **Difficulté** : Similaire

---

## 📥 Téléchargement des données

### Méthode 1 : Kaggle API (Recommandé)

```bash
# Installer Kaggle API
pip install kaggle

# Configurer les identifiants (kaggle.json dans ~/.kaggle/)

# Télécharger le dataset
kaggle datasets download -d hasibalmuzdadid/global-air-pollution-dataset

# Ou utiliser le script fourni
python download_kaggle_air_data.py
```

### Méthode 2 : Téléchargement Manuel

1. Aller sur https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset
2. Cliquer sur "Download" (nécessite un compte Kaggle gratuit)
3. Extraire le fichier ZIP
4. Placer le CSV dans le dossier `data/`

### Méthode 3 : Depuis Python

```python
import pandas as pd

# Si déjà téléchargé
df = pd.read_csv('data/global air pollution dataset.csv')

# Aperçu
print(df.head())
print(df.info())
```

---

## 🔍 Exploration rapide

### Commandes utiles

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_csv('data/global air pollution dataset.csv')

# Aperçu
print(df.head())
print(df.info())
print(df.describe())

# Vérifier les valeurs manquantes
print(df.isnull().sum())

# Distribution de l'AQI
plt.figure(figsize=(10, 6))
plt.hist(df['AQI Value'], bins=50, edgecolor='black')
plt.xlabel('AQI Value')
plt.ylabel('Nombre de villes')
plt.title('Distribution de l\'AQI Mondial')
plt.show()

# Distribution des catégories
print(df['AQI Category'].value_counts())

# Top 10 villes les plus polluées
print(df.nlargest(10, 'AQI Value')[['Country', 'City', 'AQI Value', 'AQI Category']])

# Top 10 pays (moyenne AQI)
print(df.groupby('Country')['AQI Value'].mean().nlargest(10))
```

---

## 📚 Références

### Articles Scientifiques

- WHO Global Air Quality Guidelines (2021)
- EPA Air Quality Index Technical Assistance Document
- Health Effects Institute - State of Global Air Reports

### Ressources en Ligne

- **OMS** : https://www.who.int/health-topics/air-pollution
- **EPA AQI** : https://www.airnow.gov/aqi/aqi-basics/
- **European Environment Agency** : https://www.eea.europa.eu/themes/air
- **Airparif** (France) : https://www.airparif.asso.fr/

---

## ⚖️ Licence

Ce dataset est disponible sous licence ouverte sur Kaggle.

Vous êtes libre de :
- ✅ Utiliser pour l'éducation et la recherche
- ✅ Partager et redistribuer
- ✅ Adapter et créer des dérivés

Sous les conditions suivantes :
- 📝 Citer la source (Hasib Al Muzdadid, Kaggle)

---

## 💡 Conseils d'utilisation

### Pour les étudiants

1. **Prenez le temps d'explorer** les données avant de modéliser
2. **Contextualisez** : Pensez aux implications de santé publique
3. **Visualisez** : Créez des cartes, graphiques pour comprendre
4. **Comparez** : Analysez les différences entre pays/régions
5. **Interprétez** : Reliez vos résultats aux enjeux environnementaux

### Pour les formateurs

1. **Contextualisez** : Reliez au cours sur l'environnement/santé
2. **Sensibilisez** : Utilisez des chiffres OMS pour l'impact
3. **Encouragez** l'exploration au-delà des consignes
4. **Reliez** aux actualités (pics de pollution, COP, etc.)
5. **Valorisez** le travail : "Vous contribuez à un enjeu majeur"

---

## 🔗 Liens utiles

- **Kaggle Dataset** : https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset
- **WHO Air Quality** : https://www.who.int/health-topics/air-pollution
- **EPA AQI** : https://www.airnow.gov/
- **OpenAQ** : https://openaq.org/
- **IQAir** : https://www.iqair.com/world-air-quality

---

**Bon travail avec les données environnementales ! 🌍**

