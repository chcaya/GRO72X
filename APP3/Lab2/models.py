# GRO722 Laboratoire 2
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Seq2seq(nn.Module):
    def __init__(self, n_hidden, n_layers, int2symb, symb2int, dict_size, device, max_len):
        super(Seq2seq, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        # Définition des couches du rnn
        self.fr_embedding = nn.Embedding(self.dict_size['fr'], n_hidden)
        self.en_embedding = nn.Embedding(self.dict_size['en'], n_hidden)
        self.encoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, self.dict_size['en'])
        self.to(device)
        
    def encoder(self, x):
        # Encodeur

        # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------

        # 1. 'x' (entrée) a une forme de (batch_size, seq_len)
        # 2. 'embedded' (plongement lexical) aura la forme (batch_size, seq_len, n_hidden)
        embedded = self.fr_embedding(x)
        
        # 3. 'out' (sorties de l'encodeur) aura la forme (batch_size, seq_len, n_hidden)
        # 4. 'hidden' (état caché final) aura la forme (n_layers, batch_size, n_hidden)
        #    Le 'hidden' initial est à zéro par défaut.
        out, hidden = self.encoder_layer(embedded)
        
        # ---------------------- Laboratoire 2 - Question 3 - Fin de la section à compléter -----------------

        return out, hidden

    
    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.max_len['en'] # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage 
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['en'])).to(self.device) # Vecteur de sortie du décodage

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):

            # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------   
            
            # 1. Plongement lexical de l'entrée (le mot prédit à l'étape précédente)
            #    'vec_in' (entrée) a la forme (batch_size, 1)
            #    'embedded' aura la forme (batch_size, 1, n_hidden)
            embedded = self.en_embedding(vec_in)
            
            # 2. Passe avant dans la couche GRU du décodeur
            #    'rnn_out' aura la forme (batch_size, 1, n_hidden)
            #    'hidden' (le nouvel état caché) aura la forme (n_layers, batch_size, n_hidden)
            rnn_out, hidden = self.decoder_layer(embedded, hidden)
            
            # 3. Passe avant dans la couche linéaire (scores pour chaque mot du dictionnaire)
            #    'output' (logits) aura la forme (batch_size, 1, dict_size['en'])
            output = self.fc(rnn_out)
            
            # 4. Stocker les logits de cette étape dans le tenseur de sortie
            vec_out[:, i, :] = output.squeeze(1) # squeeze() enlève la dimension de longueur 1
            
            # 5. Déterminer le mot prédit (le mot avec le score le plus élevé)
            #    'top_word' aura la forme (batch_size, 1)
            top_word = output.argmax(2)
            
            # 6. Le mot prédit devient l'entrée pour la prochaine itération
            vec_in = top_word

            # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------

        return vec_out, hidden, None

    def forward(self, x):
        # Passant avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out,h)
        return out, hidden, attn


class Seq2seq_attn(nn.Module):
    def __init__(self, n_hidden, n_layers, int2symb, symb2int, dict_size, device, max_len):
        super(Seq2seq_attn, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        # Définition des couches du rnn
        self.fr_embedding = nn.Embedding(self.dict_size['fr'], n_hidden)
        self.en_embedding = nn.Embedding(self.dict_size['en'], n_hidden)
        self.encoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(2*n_hidden, n_hidden, n_layers, batch_first=True)

        # Définition de la couche dense pour l'attention
        self.att_combine = nn.Linear(2*n_hidden, n_hidden)
        self.hidden2query = nn.Linear(n_hidden, n_hidden)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, self.dict_size['en'])
        self.to(device)
        
    def encoder(self, x):
        #Encodeur

        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------
        
        # Identique à la Q3
        # 1. Plongement lexical (batch_size, seq_len, n_hidden)
        embedded = self.fr_embedding(x)
        
        # 2. Couche GRU
        #    out: (batch_size, seq_len, n_hidden)
        #    hidden: (n_layers, batch_size, n_hidden)
        out, hidden = self.encoder_layer(embedded)
        
        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        return out, hidden

    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query)

        # Attention

        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        # 1. Calcul des scores d'attention
        # (batch, 1, n_hidden) @ (batch, n_hidden, seq_len) -> (batch, 1, seq_len)
        scores = torch.bmm(query, values.transpose(1, 2))
        
        # 2. Normalisation des scores avec Softmax
        # Appliqué sur la dimension de la séquence d'entrée (dim=2)
        attention_weights = F.softmax(scores, dim=2)
        
        # 3. Calcul du vecteur de contexte
        # (batch, 1, seq_len) @ (batch, seq_len, n_hidden) -> (batch, 1, n_hidden)
        attention_output = torch.bmm(attention_weights, values)

        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        return attention_output, attention_weights

    def decoderWithAttn(self, encoder_outs, hidden):
        # Décodeur avec attention

        # Initialisation des variables
        max_len = self.max_len['en'] # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage 
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['en'])).to(self.device) # Vecteur de sortie du décodage
        attention_weights = torch.zeros((batch_size, self.max_len['fr'], self.max_len['en'])).to(self.device) # Poids d'attention

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):

            # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------
            
            # 1. Plongement lexical de l'entrée
            # 'embedded' a la forme (batch_size, 1, n_hidden)
            embedded = self.en_embedding(vec_in)

            # 2. Calcul de l'attention
            # L'état caché 'hidden' (de la couche supérieure) est la "query"
            # 'query' a la forme (batch_size, 1, n_hidden)
            query = hidden[-1].unsqueeze(1) # Prend la dernière couche et ajoute une dimension
            
            # 'context' (batch, 1, n_hidden), 'attn_w' (batch, 1, seq_len_fr)
            context, attn_w = self.attentionModule(query, encoder_outs)
            
            # Stocker les poids d'attention (pour visualisation)
            # squeeze() enlève les dimensions superflues
            attention_weights[:, :, i] = attn_w.squeeze(1)

            # 3. Combiner le mot d'entrée (embedded) et le contexte
            # (batch, 1, n_hidden) + (batch, 1, n_hidden) -> (batch, 1, 2*n_hidden)
            rnn_input = torch.cat((embedded, context), dim=2)
            
            # 4. Passe avant dans la couche GRU du décodeur
            # rnn_out: (batch, 1, n_hidden)
            # hidden: (n_layers, batch, n_hidden)
            rnn_out, hidden = self.decoder_layer(rnn_input, hidden)
            
            # 5. Couche linéaire de sortie (Logits)
            # output: (batch, 1, dict_size['en'])
            output = self.fc(rnn_out)
            
            # 6. Stocker les logits
            vec_out[:, i, :] = output.squeeze(1)
            
            # 7. Le mot prédit devient l'entrée pour la prochaine itération
            top_word = output.argmax(2)
            vec_in = top_word

            # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        return vec_out, hidden, attention_weights


    def forward(self, x):
        # Passe avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoderWithAttn(out,h)
        return out, hidden, attn
