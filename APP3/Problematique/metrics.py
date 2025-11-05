# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hiver 2021
import numpy as np

def edit_distance(x,y):
    # Calcul de la distance d'édition
    
    # 1. Initialiser les longueurs
    m = len(x)
    n = len(y)
    
    # 2. Créer la matrice de coûts (m+1) x (n+1)
    dp_matrix = np.zeros((m + 1, n + 1), dtype=int)

    # 3. Initialiser la première ligne et la première colonne
    # Coût des suppressions pour transformer x en chaîne vide
    for i in range(m + 1):
        dp_matrix[i, 0] = i
    # Coût des insertions pour transformer une chaîne vide en y
    for j in range(n + 1):
        dp_matrix[0, j] = j

    # 4. Remplir le reste de la matrice
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Si les caractères sont identiques, le coût de substitution est 0
            cost = 0 if x[i - 1] == y[j - 1] else 1 # 1 si différents
            
            # Calculer le coût pour chaque opération
            deletion = dp_matrix[i - 1, j] + 1
            insertion = dp_matrix[i, j - 1] + 1
            substitution = dp_matrix[i - 1, j - 1] + cost
            
            # Le coût de la cellule est le minimum de ces trois opérations
            dp_matrix[i, j] = min(deletion, insertion, substitution)

    # 5. La distance finale est dans le coin inférieur droit
    return dp_matrix[m, n]

def confusion_matrix(true, pred, ignore=[]):
    # Calcul de la matrice de confusion

    # 1. Trouver toutes les classes uniques dans les données réelles et prédites
    all_labels = np.unique(np.concatenate((true, pred)))
    
    # 2. Filtrer les classes à ignorer
    classes = [label for label in all_labels if label not in ignore]
    num_classes = len(classes)
    
    # 3. Créer une table de correspondance (map) de label -> index
    label_to_index = {label: i for i, label in enumerate(classes)}
    
    # 4. Initialiser la matrice de confusion k x k (où k = nb de classes)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # 5. Parcourir toutes les prédictions
    for t, p in zip(true, pred):
        # Ignorer la paire si l'une des étiquettes est dans la liste 'ignore'
        if t in ignore or p in ignore:
            continue
            
        # Obtenir les indices de la matrice pour la vraie valeur et la prédiction
        true_idx = label_to_index[t]
        pred_idx = label_to_index[p]
        
        # Incrémenter la cellule [ligne_réelle, colonne_prédite]
        matrix[true_idx, pred_idx] += 1

    return matrix
