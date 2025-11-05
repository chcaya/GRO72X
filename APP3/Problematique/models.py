# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hiver 2021

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen

        self.input_dim = 2

        # Definition des couches
        # Couches pour rnn
        # Encodeur : Un GRU qui lit la séquence de trajectoire
        # input_size=2 car nous lisons des données (x, y)
        self.encoder_rnn = nn.GRU(
            input_size=self.input_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            batch_first=True
        )

        # Décodeur : Un GRU qui génère la séquence de caractères
        # L'entrée du décodeur est la concaténation de l'embedding du mot précédent (hidden_dim)
        # et du vecteur de contexte de l'attention (hidden_dim)
        self.decoder_rnn = nn.GRU(
            input_size=hidden_dim * 2, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            batch_first=True
        )

        # Plongement lexical (Embedding) pour les caractères de sortie
        self.embedding = nn.Embedding(dict_size, hidden_dim)

        # Couches pour attention
        # Couche pour transformer l'état caché du décodeur en "query"
        self.hidden2query = nn.Linear(hidden_dim, hidden_dim)

        # Couche dense pour la sortie
        # Couche linéaire finale pour transformer l'état caché du décodeur
        # en un score (logit) pour chaque mot du vocabulaire
        self.fc_out = nn.Linear(hidden_dim, dict_size)

    def forward(self, x):
        # x: (batch_size, seq_len_in, input_dim) - par ex. (B, 150, 2)
        
        # 1. Encodeur
        # encoder_outputs: (B, seq_len_in, hidden_dim)
        # hidden: (n_layers, B, hidden_dim)
        encoder_outputs, hidden = self.encoder_rnn(x)
        
        # Initialisations pour la boucle du décodeur
        batch_size = x.size(0)
        
        # Premier jeton d'entrée pour le décodeur : <sos>
        # Forme : (B, 1)
        decoder_input = torch.full((batch_size, 1), 
                                   0, # Start_Symbol to int
                                   dtype=torch.long, 
                                   device=self.device)
        
        # L'état caché initial du décodeur est l'état caché final de l'encodeur
        decoder_hidden = hidden
        
        # Liste pour stocker les sorties (logits) à chaque pas de temps
        outputs = []

        # 2. Boucle du décodeur (génération)
        # Nous générons un mot à la fois, jusqu'à la longueur maximale
        for t in range(self.maxlen):
            
            # 2a. Plongement lexical du mot d'entrée
            # embedded: (B, 1, hidden_dim)
            embedded = self.embedding(decoder_input)
            
            # 2b. Calcul de l'attention
            # 'query' est basé sur la couche supérieure de l'état caché du décodeur
            # query: (B, 1, hidden_dim)
            query = self.hidden2query(decoder_hidden[-1].unsqueeze(1))
            
            # Calcul des scores (dot product attention)
            # (B, 1, L_in)
            scores = torch.bmm(query, encoder_outputs.transpose(1, 2))
            
            # Poids d'attention (B, 1, L_in)
            attn_weights = F.softmax(scores, dim=2)
            
            # Vecteur de contexte (B, 1, hidden_dim)
            context = torch.bmm(attn_weights, encoder_outputs)
            
            # 2c. Préparation de l'entrée du décodeur RNN
            # Concaténation de l'embedding du mot et du contexte
            # rnn_input: (B, 1, hidden_dim * 2)
            rnn_input = torch.cat((embedded, context), dim=2)
            
            # 2d. Passe avant dans le GRU du décodeur
            # rnn_output: (B, 1, hidden_dim)
            rnn_output, decoder_hidden = self.decoder_rnn(rnn_input, decoder_hidden)
            
            # 2e. Calcul des logits de sortie
            # output: (B, dict_size)
            output = self.fc_out(rnn_output.squeeze(1))
            
            # Stocker la sortie
            outputs.append(output)
            
            # 2f. Préparation de la prochaine entrée
            # Le mot prédit (argmax) devient l'entrée de l'étape suivante
            # vec_in: (B, 1)
            decoder_input = output.argmax(dim=1).unsqueeze(1)

        # 3. Empiler les sorties
        # Convertit la liste de tenseurs (B, dict_size) en un seul tenseur
        # (B, maxlen, dict_size)
        outputs = torch.stack(outputs, dim=1)
        
        return outputs
