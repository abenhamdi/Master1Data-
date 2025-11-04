"""
Script de génération de données synthétiques pour le TP Système de Recommandation
Ce script crée des données réalistes d'e-commerce pour l'apprentissage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuration pour la reproductibilité
np.random.seed(42)
random.seed(42)

# Paramètres de génération
N_USERS = 5000
N_PRODUCTS = 1000
N_INTERACTIONS = 50000

print("🔧 Génération des données synthétiques en cours...")
print(f"   Utilisateurs : {N_USERS}")
print(f"   Produits : {N_PRODUCTS}")
print(f"   Interactions : {N_INTERACTIONS}")
print()

# ============================================================================
# 1. GÉNÉRATION DES UTILISATEURS
# ============================================================================
print("👥 Génération des utilisateurs...")

# Âges avec distribution réaliste (18-70 ans, pic à 25-40)
ages = np.concatenate([
    np.random.normal(30, 8, N_USERS // 2),  # Jeunes adultes
    np.random.normal(50, 10, N_USERS // 2)  # Adultes plus âgés
])
ages = np.clip(ages, 18, 70).astype(int)

# Genres
genders = np.random.choice(['M', 'F', 'Other'], N_USERS, p=[0.48, 0.48, 0.04])

# Localisation (départements français)
departments = ['75', '13', '69', '31', '44', '33', '59', '34', '35', '67']
locations = np.random.choice(departments, N_USERS, p=[0.20, 0.10, 0.10, 0.08, 0.07, 0.07, 0.08, 0.08, 0.07, 0.15])

# Date d'inscription (sur les 2 dernières années)
start_date = datetime.now() - timedelta(days=730)
registration_dates = [start_date + timedelta(days=random.randint(0, 730)) for _ in range(N_USERS)]

# Niveau d'activité (1=faible, 5=très actif)
activity_level = np.random.choice([1, 2, 3, 4, 5], N_USERS, p=[0.30, 0.25, 0.25, 0.15, 0.05])

users_df = pd.DataFrame({
    'user_id': range(1, N_USERS + 1),
    'age': ages,
    'gender': genders,
    'location': locations,
    'registration_date': registration_dates,
    'activity_level': activity_level
})

print(f"   ✅ {len(users_df)} utilisateurs créés")

# ============================================================================
# 2. GÉNÉRATION DES PRODUITS
# ============================================================================
print("📦 Génération des produits...")

# Catégories de produits tech
categories = [
    'Ordinateurs', 'Smartphones', 'Tablettes', 'Accessoires', 
    'Audio', 'Photo', 'Gaming', 'Wearables', 'Smart Home', 'Composants'
]

# Sous-catégories par catégorie
subcategories_map = {
    'Ordinateurs': ['Laptop', 'Desktop', 'Ultrabook', 'Workstation'],
    'Smartphones': ['Android', 'iPhone', 'Feature Phone'],
    'Tablettes': ['iPad', 'Android Tablet', 'Windows Tablet'],
    'Accessoires': ['Clavier', 'Souris', 'Casque', 'Housse', 'Cable'],
    'Audio': ['Casque Bluetooth', 'Enceinte', 'Écouteurs', 'Soundbar'],
    'Photo': ['Appareil Photo', 'Objectif', 'Drone', 'Stabilisateur'],
    'Gaming': ['Console', 'Jeux', 'Manette', 'PC Gaming'],
    'Wearables': ['Smartwatch', 'Bracelet Connecté', 'Lunettes AR'],
    'Smart Home': ['Caméra', 'Thermostat', 'Alarme', 'Éclairage'],
    'Composants': ['Processeur', 'Carte Graphique', 'RAM', 'SSD']
}

product_categories = np.random.choice(categories, N_PRODUCTS, p=[0.15, 0.15, 0.08, 0.18, 0.10, 0.08, 0.12, 0.05, 0.06, 0.03])
product_subcategories = [random.choice(subcategories_map[cat]) for cat in product_categories]

# Noms de produits réalistes
brands = ['Apple', 'Samsung', 'Lenovo', 'HP', 'Dell', 'Sony', 'LG', 'Xiaomi', 'Asus', 'Acer']
product_names = [
    f"{random.choice(brands)} {cat} {subcat} {random.choice(['Pro', 'Plus', 'Max', 'Lite', 'Standard'])}"
    for cat, subcat in zip(product_categories, product_subcategories)
]

# Prix avec distribution réaliste par catégorie
price_ranges = {
    'Ordinateurs': (500, 3000),
    'Smartphones': (200, 1500),
    'Tablettes': (150, 1200),
    'Accessoires': (10, 200),
    'Audio': (30, 500),
    'Photo': (300, 5000),
    'Gaming': (200, 2000),
    'Wearables': (100, 800),
    'Smart Home': (50, 500),
    'Composants': (50, 2000)
}

prices = [
    round(np.random.uniform(price_ranges[cat][0], price_ranges[cat][1]), 2)
    for cat in product_categories
]

# Stock disponible
stock = np.random.randint(0, 500, N_PRODUCTS)

# Note moyenne initiale (avant interactions)
initial_rating = np.random.uniform(3.5, 5.0, N_PRODUCTS).round(1)

# Date d'ajout au catalogue
product_add_dates = [start_date + timedelta(days=random.randint(0, 700)) for _ in range(N_PRODUCTS)]

products_df = pd.DataFrame({
    'product_id': range(1, N_PRODUCTS + 1),
    'name': product_names,
    'category': product_categories,
    'subcategory': product_subcategories,
    'price': prices,
    'stock': stock,
    'initial_rating': initial_rating,
    'added_date': product_add_dates
})

print(f"   ✅ {len(products_df)} produits créés")

# ============================================================================
# 3. GÉNÉRATION DES INTERACTIONS
# ============================================================================
print("🔗 Génération des interactions...")

# Types d'interactions
interaction_types = ['view', 'add_to_cart', 'purchase', 'review']

# Générer les interactions avec des patterns réalistes
interactions_list = []

# Créer une matrice de préférences utilisateur-catégorie
user_category_preference = {}
for user_id in range(1, N_USERS + 1):
    # Chaque utilisateur a 2-3 catégories préférées
    n_preferred = random.randint(2, 3)
    preferred_cats = random.sample(categories, n_preferred)
    user_category_preference[user_id] = preferred_cats

for _ in range(N_INTERACTIONS):
    # Sélectionner un utilisateur (plus actif = plus de chances)
    user_id = random.choices(
        users_df['user_id'].tolist(),
        weights=users_df['activity_level'].tolist()
    )[0]
    
    # Sélectionner un produit (favoriser les catégories préférées)
    user_prefs = user_category_preference[user_id]
    
    # 70% de chances de choisir dans les catégories préférées
    if random.random() < 0.7:
        preferred_products = products_df[products_df['category'].isin(user_prefs)]
        if len(preferred_products) > 0:
            product = preferred_products.sample(1).iloc[0]
        else:
            product = products_df.sample(1).iloc[0]
    else:
        product = products_df.sample(1).iloc[0]
    
    product_id = product['product_id']
    
    # Type d'interaction (funnel réaliste)
    # view (70%) -> add_to_cart (20%) -> purchase (8%) -> review (2%)
    interaction_type = np.random.choice(
        interaction_types,
        p=[0.70, 0.20, 0.08, 0.02]
    )
    
    # Date d'interaction (entre l'inscription de l'user et aujourd'hui)
    user_reg_date = users_df[users_df['user_id'] == user_id]['registration_date'].iloc[0]
    product_add_date = product['added_date']
    
    # L'interaction doit être après les deux dates
    min_date = max(user_reg_date, product_add_date)
    max_date = datetime.now()
    
    if min_date < max_date:
        days_diff = (max_date - min_date).days
        interaction_date = min_date + timedelta(days=random.randint(0, days_diff))
    else:
        interaction_date = max_date
    
    # Rating (seulement pour purchase et review)
    if interaction_type in ['purchase', 'review']:
        # Distribution réaliste des notes (plus de 4-5 étoiles)
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.05, 0.15, 0.35, 0.40])
    else:
        rating = None
    
    # Durée de session (en secondes) pour les views
    if interaction_type == 'view':
        session_duration = int(np.random.exponential(60))  # Moyenne 60 secondes
    else:
        session_duration = None
    
    interactions_list.append({
        'interaction_id': len(interactions_list) + 1,
        'user_id': user_id,
        'product_id': product_id,
        'interaction_type': interaction_type,
        'interaction_date': interaction_date,
        'rating': rating,
        'session_duration': session_duration
    })

interactions_df = pd.DataFrame(interactions_list)

# Trier par date
interactions_df = interactions_df.sort_values('interaction_date').reset_index(drop=True)
interactions_df['interaction_id'] = range(1, len(interactions_df) + 1)

print(f"   ✅ {len(interactions_df)} interactions créées")

# ============================================================================
# 4. AJOUT DE DONNÉES COMPLÉMENTAIRES
# ============================================================================
print("🔧 Ajout de données complémentaires...")

# Ajouter des descriptions de produits (pour content-based filtering)
adjectives = ['Performant', 'Élégant', 'Innovant', 'Puissant', 'Compact', 'Révolutionnaire', 'Premium', 'Polyvalent']
features = ['Design moderne', 'Haute performance', 'Longue autonomie', 'Connectivité avancée', 'Interface intuitive']

descriptions = []
for idx, row in products_df.iterrows():
    desc = f"{random.choice(adjectives)} {row['name']}. "
    desc += f"{random.choice(features)}. "
    desc += f"Idéal pour {row['subcategory'].lower()}. "
    desc += f"Catégorie {row['category']}."
    descriptions.append(desc)

products_df['description'] = descriptions

# ============================================================================
# 5. STATISTIQUES ET VÉRIFICATIONS
# ============================================================================
print("\n📊 Statistiques des données générées :")
print(f"   Utilisateurs : {len(users_df)}")
print(f"   Produits : {len(products_df)}")
print(f"   Interactions : {len(interactions_df)}")
print()

print("   Distribution des interactions :")
print(interactions_df['interaction_type'].value_counts())
print()

print("   Distribution des catégories de produits :")
print(products_df['category'].value_counts().head())
print()

# ============================================================================
# 6. SAUVEGARDE DES FICHIERS CSV
# ============================================================================
print("💾 Sauvegarde des fichiers...")

users_df.to_csv('users.csv', index=False)
print("   ✅ users.csv créé")

products_df.to_csv('products.csv', index=False)
print("   ✅ products.csv créé")

interactions_df.to_csv('interactions.csv', index=False)
print("   ✅ interactions.csv créé")

# ============================================================================
# 7. CRÉATION D'UN FICHIER DE MÉTA-DONNÉES
# ============================================================================
metadata = {
    'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_users': len(users_df),
    'n_products': len(products_df),
    'n_interactions': len(interactions_df),
    'date_range': f"{interactions_df['interaction_date'].min()} to {interactions_df['interaction_date'].max()}",
    'categories': products_df['category'].unique().tolist(),
    'interaction_types': interactions_df['interaction_type'].unique().tolist()
}

import json
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

print("   ✅ metadata.json créé")

print("\n✅ Génération terminée avec succès !")
print("\n📁 Fichiers créés :")
print("   - users.csv")
print("   - products.csv")
print("   - interactions.csv")
print("   - metadata.json")
print("\n🚀 Vous pouvez maintenant commencer le TP !")

