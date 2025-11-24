#!/usr/bin/env python3
"""
Script pour télécharger le Global Air Pollution Dataset depuis Kaggle
"""

import os
import subprocess
import pandas as pd

# Configuration Kaggle
DATASET_NAME = 'hasibalmuzdadid/global-air-pollution-dataset'
DATA_DIR = 'data'

def check_kaggle_setup():
    """
    Vérifie que Kaggle API est installée et configurée
    """
    print("🔍 Vérification de la configuration Kaggle...")
    
    try:
        import kaggle
        print("✅ Kaggle API installée")
        return True
    except ImportError:
        print("❌ Kaggle API non installée")
        print("\n📦 Pour installer Kaggle API :")
        print("   pip install kaggle")
        print("\n🔑 Pour configurer vos identifiants :")
        print("   1. Aller sur https://www.kaggle.com/settings")
        print("   2. Cliquer sur 'Create New API Token'")
        print("   3. Placer le fichier kaggle.json dans ~/.kaggle/")
        print("   4. Sur Linux/Mac: chmod 600 ~/.kaggle/kaggle.json")
        return False

def download_dataset():
    """
    Télécharge le dataset depuis Kaggle
    """
    print("\n🌍 Téléchargement du Global Air Pollution Dataset...")
    print("=" * 60)
    
    # Créer le dossier data
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"✅ Dossier '{DATA_DIR}/' créé")
    
    try:
        # Télécharger avec Kaggle API
        print(f"\n📥 Téléchargement depuis Kaggle...")
        cmd = f"kaggle datasets download -d {DATASET_NAME} -p {DATA_DIR} --unzip"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dataset téléchargé et extrait avec succès !")
            return True
        else:
            print(f"❌ Erreur lors du téléchargement")
            print(f"   {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return False

def load_and_preview_data():
    """
    Charge et affiche un aperçu des données
    """
    print("\n" + "=" * 60)
    print("🔍 Chargement et aperçu des données...")
    print("=" * 60)
    
    try:
        # Chercher le fichier CSV
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        
        if not csv_files:
            print("❌ Aucun fichier CSV trouvé")
            return False
        
        # Charger le premier fichier CSV trouvé
        csv_file = os.path.join(DATA_DIR, csv_files[0])
        print(f"\n📊 Fichier trouvé : {csv_files[0]}")
        
        df = pd.read_csv(csv_file)
        
        print(f"\n✅ Données chargées avec succès !")
        print(f"   Observations : {len(df):,}")
        print(f"   Variables    : {len(df.columns)}")
        
        print(f"\n📋 Colonnes disponibles :")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        print(f"\n📊 Aperçu des données (5 premières lignes) :")
        print(df.head())
        
        print(f"\n📈 Statistiques descriptives :")
        print(df.describe())
        
        print(f"\n⚠️ Valeurs manquantes :")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("   ✅ Aucune valeur manquante")
        
        # Informations spécifiques au dataset
        if 'AQI Value' in df.columns:
            print(f"\n🌍 Statistiques AQI (Air Quality Index) :")
            print(f"   Minimum : {df['AQI Value'].min():.2f}")
            print(f"   Maximum : {df['AQI Value'].max():.2f}")
            print(f"   Moyenne : {df['AQI Value'].mean():.2f}")
            print(f"   Médiane : {df['AQI Value'].median():.2f}")
        
        if 'AQI Category' in df.columns:
            print(f"\n📊 Distribution des catégories de qualité d'air :")
            print(df['AQI Category'].value_counts())
        
        if 'Country' in df.columns:
            print(f"\n🌐 Nombre de pays : {df['Country'].nunique()}")
            print(f"   Top 5 pays (nombre de villes) :")
            print(df['Country'].value_counts().head())
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return False

def create_info_file():
    """
    Crée un fichier d'information sur le dataset
    """
    info_content = """# Global Air Pollution Dataset - Informations

## Source
Kaggle - Global Air Pollution Dataset
URL: https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset
Auteur: Hasib Al Muzdadid

## Description
Ce dataset contient des données sur la qualité de l'air pour plus de 23,000 villes
dans le monde, collectées entre 2017 et 2022.

## Période
2017 - 2022 (données récentes)

## Couverture
Plus de 23,000 villes dans le monde

## Variables principales

### Polluants mesurés
- **AQI Value** : Indice de Qualité de l'Air (0-500)
- **AQI Category** : Catégorie (Good, Moderate, Unhealthy, etc.)
- **CO AQI Value** : Monoxyde de carbone
- **Ozone AQI Value** : Ozone (O3)
- **NO2 AQI Value** : Dioxyde d'azote
- **PM2.5 AQI Value** : Particules fines (<2.5 μm)

### Informations géographiques
- **Country** : Pays
- **City** : Ville
- **Latitude / Longitude** : Coordonnées GPS

## Catégories AQI (Air Quality Index)

- **Good** (0-50) : Qualité de l'air satisfaisante
- **Moderate** (51-100) : Acceptable, mais risque pour personnes sensibles
- **Unhealthy for Sensitive Groups** (101-150) : Risque pour groupes sensibles
- **Unhealthy** (151-200) : Risque pour toute la population
- **Very Unhealthy** (201-300) : Alerte sanitaire
- **Hazardous** (301+) : Urgence sanitaire

## Utilisation pédagogique

Ce dataset est idéal pour :
- Classification de la qualité de l'air (bon/mauvais)
- Prédiction de l'AQI
- Analyse comparative entre pays/villes
- Visualisation géographique de la pollution
- Sensibilisation aux enjeux environnementaux
- Analyse de l'impact de la pollution sur la santé

## Impact environnemental

Selon l'OMS :
- 99% de la population mondiale respire un air pollué
- 7 millions de décès prématurés par an dus à la pollution
- Principal risque environnemental pour la santé

## Contexte d'utilisation

Ce TD permet aux étudiants de :
- Travailler sur des données environnementales réelles et récentes
- Comprendre les enjeux de santé publique
- Appliquer le ML à un problème sociétal important
- Contribuer à la sensibilisation environnementale

## Citation

Si vous utilisez ce dataset, merci de citer :
Hasib Al Muzdadid (2022). Global Air Pollution Dataset. Kaggle.
https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset
"""
    
    info_file = os.path.join(DATA_DIR, 'DATASET_INFO.txt')
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(info_content)
    
    print(f"\n📄 Fichier d'information créé : {info_file}")

def main():
    """
    Fonction principale
    """
    print("\n" + "🌍" * 30)
    print("   TÉLÉCHARGEMENT DU GLOBAL AIR POLLUTION DATASET")
    print("🌍" * 30 + "\n")
    
    # Vérifier la configuration Kaggle
    if not check_kaggle_setup():
        print("\n⚠️ Veuillez configurer Kaggle API avant de continuer")
        print("\n💡 Alternative : Téléchargement manuel")
        print("   1. Aller sur : https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset")
        print("   2. Cliquer sur 'Download'")
        print(f"   3. Extraire le fichier dans le dossier '{DATA_DIR}/'")
        return
    
    # Télécharger le dataset
    if not download_dataset():
        print("\n❌ Échec du téléchargement")
        return
    
    # Charger et afficher un aperçu
    if not load_and_preview_data():
        print("\n⚠️ Impossible d'afficher l'aperçu")
    
    # Créer le fichier d'information
    create_info_file()
    
    print("\n" + "=" * 60)
    print("🎉 Tout est prêt pour le TD !")
    print("=" * 60)
    print("\n💡 Vous pouvez maintenant charger les données avec :")
    print("   import pandas as pd")
    print("   df = pd.read_csv('data/global air pollution dataset.csv')")
    print("\n🌍 Bon travail sur ce projet environnemental !")
    print("   Votre travail contribue à la sensibilisation aux enjeux de santé publique.\n")

if __name__ == '__main__':
    main()

