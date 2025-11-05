import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        ### Extraction des symboles ###
        # 1. Initialiser le vocabulaire avec les jetons spéciaux
        self.symb2int = {self.start_symbol: 0, self.stop_symbol: 1, self.pad_symbol: 2}
        char_symb = 3
        
        # 2. Construire le vocabulaire à partir de tous les caractères cibles (mots)
        for word, _ in self.data:
            for char in word:
                if char not in self.symb2int:
                    self.symb2int[char] = char_symb
                    char_symb += 1
                    
        # 3. Créer le dictionnaire inverse (entier -> symbole)
        self.int2symb = {v: k for k, v in self.symb2int.items()}
        self.dict_size = len(self.symb2int)


        ### Ajout du padding aux séquences ###
        self.max_len_in = 0
        self.max_len_target = 0
        
        # 4. Trouver les longueurs maximales pour les entrées (strokes) et les cibles (mots)
        # Note : stroke_array.shape[1] est la longueur de la séquence (L)
        self.max_len_in = max(stroke_array.shape[1] for _, stroke_array in self.data)
        self.max_len_target = max(len(word) for word, _ in self.data) + 1 # +1 pour <eos>
        print(f'self.max_len_in: {self.max_len_in}')
        print(f'self.max_len_target: {self.max_len_target}')
        
        # 5. Déterminer la dimension des features d'entrée (nous prenons X et Y, donc 2)
        stroke_feature_dim = self.data[0][1].shape[0]
        
        # 6. Appliquer le padding et la tokenisation
        for i in range(len(self.data)):
            word, strokes = self.data[i]
            
            # --- Remplissage (Padding) de la cible (mot) ---
            char_list = list(word)
            char_list.append(self.stop_symbol) # Ajouter <eos>
            
            n_pad_target = self.max_len_target - len(char_list)
            char_list.extend([self.pad_symbol] * n_pad_target)
            
            # Convertir les caractères en entiers (tokenisation)
            tokenized_word = [self.symb2int[char] for char in char_list]
            padded_word = np.array(tokenized_word)
            
            # --- Remplissage (Padding) de l'entrée (strokes) ---
            # Prendre uniquement les 2 premières features (X, Y)
            xy_strokes = strokes[0:2, :] # Forme (2, L)
            current_len = xy_strokes.shape[1]
            n_pad_in = self.max_len_in - current_len
            
            # Créer un array de zéros pour le padding
            pad_array_in = np.zeros((stroke_feature_dim, n_pad_in))
            
            # Concaténer horizontalement (le long de la dimension de la séquence)
            padded_strokes = np.hstack([xy_strokes, pad_array_in]) # Forme (2, max_len_in)
            
            # Transposer pour avoir le format (longueur de séquence, nb de features)
            padded_strokes = padded_strokes.T # Forme (max_len_in, 2)
            
            # Remplacer l'échantillon original par l'échantillon traité
            self.data[i] = (padded_strokes, padded_word)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Récupérer l'échantillon pré-traité (rembourré et tokenisé)
        input_seq, target_seq = self.data[idx]
        
        # Convertir les arrays numpy en Tenseurs PyTorch
        # Les "strokes" (entrées) sont des flottants
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)
        # Les indices de caractères (cibles) sont des entiers (long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        
        return input_tensor, target_tensor

    def visualisation(self, idx):
        # Visualisation des échantillons
        input_seq_tensor, target_seq_tensor = self[idx]
        
        # Reconvertir en numpy pour l'affichage
        input_seq = input_seq_tensor.numpy()
        target_seq = target_seq_tensor.numpy()
        
        # Reconstruire le mot à partir des jetons (tokens)
        word_chars = []
        for token in target_seq:
            char = self.int2symb[token]
            if char == self.stop_symbol or char == self.pad_symbol:
                break # Arrêter au premier jeton de fin ou de padding
            word_chars.append(char)
        word = "".join(word_chars)

        # 1. Trouver la longueur originale avant le padding
        #    input_seq a la forme (max_longueur, 2)
        #    Nous trouvons le premier index où la ligne est [0, 0]
        
        #    np.all(axis=1) trouve les lignes où les deux (x,y) sont 0
        zero_rows_mask = np.all(input_seq == 0, axis=1)
        
        #    np.argmax() trouvera le *premier* index True,
        #    qui est le début de la zone de padding.
        first_pad_index = np.argmax(zero_rows_mask)
        
        # 2. Gérer le cas où il n'y a pas de padding
        #    Si first_pad_index est 0, cela peut signifier "pas de padding"
        #    ou "padding dès le début". Nous vérifions le premier point.
        if first_pad_index == 0 and not np.all(input_seq[0] == 0):
            original_len = len(input_seq) # Pas de padding trouvé
        else:
            original_len = first_pad_index # On a trouvé l'index du padding

        # 3. Extraire seulement les données de trait originales
        original_strokes = input_seq[:original_len, :]
        
        # Afficher les "strokes" (coordonnées x, y)
        # input_seq a la forme (longueur, 2)
        plt.plot(original_strokes[:, 0], original_strokes[:, 1])
        plt.title(f"Échantillon {idx}: '{word}'")
        plt.xlabel("Coordonnée X")
        plt.ylabel("Coordonnée Y")
        plt.show()


if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))
