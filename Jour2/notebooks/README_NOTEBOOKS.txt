GUIDE DES NOTEBOOKS - TP SYSTEME DE RECOMMANDATION


Ce dossier contient 5 notebooks Jupyter pour guider votre travail.

ORDRE DE TRAVAIL


1. 00_Guide_Demarrage.ipynb
   - Installation et configuration
   - Génération des données
   - Premiers tests
   - Fonctions utilitaires
   


2. 01_EDA_starter.ipynb
   - Analyse exploratoire des données
   - Statistiques descriptives
   - Visualisations
   - Analyse du cold start


3. 02_Features_starter.ipynb
   - Feature engineering utilisateurs
   - Feature engineering produits
   - Normalisation et encodage
   - Analyse de corrélation
   

4. 03_Modeling_starter.ipynb
   - Filtrage collaboratif (User-Based)
   - Filtrage par contenu (Content-Based)
   - Modèle hybride (ML supervisé)
   - Comparaison préliminaire
   

5. 04_Evaluation_starter.ipynb
   - Implémentation des métriques (Precision@K, MAP, NDCG)
   - Évaluation des trois modèles
   - Comparaison finale
   - Métriques business
   


STRUCTURE DES NOTEBOOKS

Chaque notebook contient :

- Une introduction avec les objectifs
- Des sections numérotées correspondant au sujet
- Des TODO pour guider votre travail
- Des cellules à compléter avec "VOTRE CODE ICI"
- Des exemples et des tests


CONSEILS D'UTILISATION

1. Suivez l'ordre des notebooks
2. Ne passez au suivant que lorsque le précédent fonctionne
3. Sauvegardez régulièrement (Ctrl+S ou Cmd+S)
4. Testez votre code au fur et à mesure
5. Commentez vos choix techniques
6. Consultez l'AIDE_MEMOIRE.md en cas de blocage


ERREURS DE LINTER

Les commentaires "# VOTRE CODE ICI" génèrent des erreurs de linter.
C'est normal, elles disparaîtront quand vous compléterez le code.


SAUVEGARDE DES DONNÉES

Si vous créez des objets importants (matrices, modèles), pensez à les sauvegarder :

import pickle

# Sauvegarder
with open('../data/user_item_matrix.pkl', 'wb') as f:
    pickle.dump(user_item_matrix, f)

# Charger
with open('../data/user_item_matrix.pkl', 'rb') as f:
    user_item_matrix = pickle.load(f)


ENVIRONNEMENT RECOMMANDÉ

- Python 3.8+
- Jupyter Lab ou Jupyter Notebook
- Toutes les bibliothèques du requirements.txt installées


BESOIN D'AIDE ?

1. Relisez le sujet (SUJET_TP_Systeme_Recommandation.md)
2. Vérifiez la documentation officielle des bibliothèques


Bon travail !

