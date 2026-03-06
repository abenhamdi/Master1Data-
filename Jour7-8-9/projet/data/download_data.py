"""
Script de téléchargement du dataset Telecom Churn depuis Kaggle
Dataset: mnassrib/telecom-churn-prediction
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

def download_with_kaggle():
    """
    Télécharge le dataset via l'API Kaggle.
    Nécessite: pip install kaggle + configuration ~/.kaggle/kaggle.json
    """
    try:
        import kaggle
        print("📦 Téléchargement via Kaggle API...")
        
        # Télécharger le dataset
        kaggle.api.dataset_download_files(
            'mnassrib/telecom-churn-prediction',
            path='.',
            unzip=True
        )
        
        print("✓ Dataset téléchargé avec succès !")
        return True
        
    except ImportError:
        print("❌ Kaggle API non installée.")
        print("   Installation: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Erreur Kaggle API: {e}")
        return False

def download_manual():
    """
    Instructions pour téléchargement manuel.
    """
    print("\n" + "="*60)
    print("TÉLÉCHARGEMENT MANUEL REQUIS")
    print("="*60)
    print("\n1. Connectez-vous à Kaggle: https://www.kaggle.com/")
    print("\n2. Téléchargez le dataset:")
    print("   https://www.kaggle.com/datasets/mnassrib/telecom-churn-prediction")
    print("\n3. Décompressez l'archive dans ce dossier:")
    print(f"   {Path.cwd()}")
    print("\n4. Vérifiez la présence du fichier: telecom_churn.csv")
    print("="*60 + "\n")

def check_dataset():
    """
    Vérifie la présence du dataset.
    """
    files = ['telecom_churn.csv', 'churn.csv', 'WA_Fn-UseC_-Telco-Customer-Churn.csv']
    
    for filename in files:
        if os.path.exists(filename):
            print(f"✓ Dataset trouvé: {filename}")
            
            # Afficher les informations
            size = os.path.getsize(filename) / (1024 * 1024)
            print(f"  Taille: {size:.2f} MB")
            
            # Compter les lignes
            with open(filename, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            print(f"  Lignes: {lines:,}")
            
            return True
    
    return False

def setup_kaggle_api():
    """
    Guide pour configurer l'API Kaggle.
    """
    print("\n" + "="*60)
    print("CONFIGURATION KAGGLE API")
    print("="*60)
    print("\n1. Créez un compte Kaggle: https://www.kaggle.com/")
    print("\n2. Allez dans votre profil > Account > API")
    print("\n3. Cliquez 'Create New API Token'")
    print("   → Télécharge kaggle.json")
    print("\n4. Placez kaggle.json dans:")
    print("   • Linux/Mac: ~/.kaggle/kaggle.json")
    print("   • Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
    print("\n5. Permissions (Linux/Mac):")
    print("   chmod 600 ~/.kaggle/kaggle.json")
    print("\n6. Installez l'API:")
    print("   pip install kaggle")
    print("="*60 + "\n")

def main():
    """
    Point d'entrée principal.
    """
    print("="*60)
    print("TÉLÉCHARGEMENT DATASET TELECOM CHURN")
    print("="*60 + "\n")
    
    # Vérifier si le dataset existe déjà
    if check_dataset():
        print("\n✓ Dataset déjà présent. Téléchargement non nécessaire.")
        return
    
    print("Dataset non trouvé. Tentative de téléchargement...\n")
    
    # Méthode 1: API Kaggle
    if download_with_kaggle():
        check_dataset()
        return
    
    # Méthode 2: Instructions manuelles
    print("\n⚠️  API Kaggle non disponible.")
    
    choice = input("\nAfficher le guide de configuration Kaggle ? (o/n): ")
    if choice.lower() == 'o':
        setup_kaggle_api()
    
    download_manual()

if __name__ == "__main__":
    main()
