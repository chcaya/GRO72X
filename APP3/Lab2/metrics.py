import numpy as np
import time
    
def edit_distance(a,b):
    # Calcul de la distance d'édition

    # ---------------------- Laboratoire 2 - Question 1 - Début de la section à compléter ------------------

    # 1. Initialiser les longueurs des séquences
    m = len(a)
    n = len(b)

    # 2. Initialiser la matrice de coûts (de taille (m+1) x (n+1))
    # Nous utilisons numpy car il est déjà importé
    dp_matrix = np.zeros((m + 1, n + 1), dtype=int)

    # 3. Remplir la première ligne et la première colonne
    # Coût pour transformer une chaîne vide en 'a' (suppressions)
    for i in range(m + 1):
        dp_matrix[i, 0] = i
    # Coût pour transformer une chaîne vide en 'b' (insertions)
    for j in range(n + 1):
        dp_matrix[0, j] = j

    # 4. Remplir le reste de la matrice
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Si les caractères sont identiques, le coût de substitution est 0
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1 # Coût de substitution
            
            # Le coût est le minimum de trois opérations :
            # 1. Suppression (depuis a)
            deletion = dp_matrix[i - 1, j] + 1
            # 2. Insertion (dans a)
            insertion = dp_matrix[i, j - 1] + 1
            # 3. Substitution (ou correspondance)
            substitution = dp_matrix[i - 1, j - 1] + cost
            
            dp_matrix[i, j] = min(deletion, insertion, substitution)

    # 5. La distance finale est dans le coin inférieur droit
    return dp_matrix[m, n]
    
    # ---------------------- Laboratoire 2 - Question 1 - Fin de la section à compléter ------------------

if __name__ =="__main__":
    a = list('allo')
    b = list('apollo2')
    c = edit_distance(a,b)

    print('Distance d\'edition entre ',str(a),' et ',str(b), ': ', c)
