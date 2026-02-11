"""
Script de téléchargement du dataset Chest X-Ray Pneumonia depuis Kaggle.
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

import os
import sys
import zipfile
from pathlib import Path

def download_with_kaggle():
    """
    Télécharge le dataset via l'API Kaggle.
    Nécessite: pip install kaggle + configuration ~/.kaggle/kaggle.json
    """
    try:
        import kaggle
        print("📦 Téléchargement du dataset Chest X-Ray Pneumonia via Kaggle API...")
        
        # Télécharger le dataset
        kaggle.api.dataset_download_files(
            'paultimothymooney/chest-xray-pneumonia',
            path='.',
            unzip=True
        )
        
        print("✓ Dataset téléchargé et décompressé avec succès !")
        return True
        
    except ImportError:
        print("❌ Kaggle API non installée.")
        print("   Installation: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Erreur Kaggle API: {e}")
        return False

def check_dataset():
    """
    Vérifie la présence et la structure du dataset.
    """
    expected_structure = {
        'chest_xray': {
            'train': ['NORMAL', 'PNEUMONIA'],
            'val': ['NORMAL', 'PNEUMONIA'],
            'test': ['NORMAL', 'PNEUMONIA']
        }
    }
    
    if not os.path.exists('chest_xray'):
        print("❌ Dossier 'chest_xray' non trouvé.")
        return False
    
    # Vérifier la structure
    for split in ['train', 'val', 'test']:
        split_path = os.path.join('chest_xray', split)
        if not os.path.exists(split_path):
            print(f"❌ Dossier '{split}' manquant.")
            return False
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                print(f"❌ Dossier '{split}/{class_name}' manquant.")
                return False
            
            # Compter les images
            num_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpeg', '.jpg', '.png'))])
            print(f"✓ {split}/{class_name}: {num_images} images")
    
    print("\n✓ Structure du dataset validée !")
    return True

def get_dataset_stats():
    """
    Affiche les statistiques du dataset.
    """
    if not os.path.exists('chest_xray'):
        print("❌ Dataset non trouvé. Exécutez d'abord le téléchargement.")
        return
    
    print("\n" + "="*60)
    print("STATISTIQUES DU DATASET")
    print("="*60)
    
    total_images = 0
    for split in ['train', 'val', 'test']:
        split_path = os.path.join('chest_xray', split)
        normal_count = len(os.listdir(os.path.join(split_path, 'NORMAL')))
        pneumonia_count = len(os.listdir(os.path.join(split_path, 'PNEUMONIA')))
        split_total = normal_count + pneumonia_count
        total_images += split_total
        
        print(f"\n{split.upper()}:")
        print(f"  NORMAL:    {normal_count:>5} images ({normal_count/split_total*100:.1f}%)")
        print(f"  PNEUMONIA: {pneumonia_count:>5} images ({pneumonia_count/split_total*100:.1f}%)")
        print(f"  TOTAL:     {split_total:>5} images")
    
    print(f"\nTOTAL DATASET: {total_images} images")
    print("="*60)

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

def download_manual():
    """
    Instructions pour téléchargement manuel.
    """
    print("\n" + "="*60)
    print("TÉLÉCHARGEMENT MANUEL")
    print("="*60)
    print("\n1. Connectez-vous à Kaggle: https://www.kaggle.com/")
    print("\n2. Téléchargez le dataset:")
    print("   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("\n3. Décompressez l'archive dans ce dossier (TD/data/)")
    print("\n4. Vérifiez la structure:")
    print("   chest_xray/")
    print("   ├── train/")
    print("   │   ├── NORMAL/")
    print("   │   └── PNEUMONIA/")
    print("   ├── val/")
    print("   │   ├── NORMAL/")
    print("   │   └── PNEUMONIA/")
    print("   └── test/")
    print("       ├── NORMAL/")
    print("       └── PNEUMONIA/")
    print("="*60 + "\n")

def main():
    """
    Point d'entrée principal.
    """
    print("="*60)
    print("TÉLÉCHARGEMENT DATASET CHEST X-RAY PNEUMONIA")
    print("="*60 + "\n")
    
    # Vérifier si le dataset existe déjà
    if check_dataset():
        print("\n✓ Dataset déjà présent et validé.")
        get_dataset_stats()
        return
    
    print("Dataset non trouvé. Téléchargement nécessaire...\n")
    
    # Méthode 1: API Kaggle
    if download_with_kaggle():
        if check_dataset():
            get_dataset_stats()
            return
    
    # Méthode 2: Instructions manuelles
    print("\n⚠️  API Kaggle non disponible.")
    
    choice = input("\nAfficher le guide de configuration Kaggle ? (o/n): ")
    if choice.lower() == 'o':
        setup_kaggle_api()
    
    download_manual()

if __name__ == "__main__":
    main()
