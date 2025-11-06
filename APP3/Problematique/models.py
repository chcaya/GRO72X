# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hiver 2021

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, max_len):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        self.input_dim = 2

        # Définition des couches du rnn
        self.embedding = nn.Embedding(dict_size, hidden_dim)
        self.encoder_rnn = nn.GRU(
            input_size=self.input_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            batch_first=True
        )
        self.decoder_rnn = nn.GRU(
            input_size=hidden_dim * 2, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            batch_first=True
        )

        # Définition de la couche dense pour l'attention
        self.att_combine = nn.Linear(2*hidden_dim, hidden_dim)
        self.hidden2query = nn.Linear(hidden_dim, hidden_dim)

        # Définition de la couche dense pour la sortie
        self.fc_out = nn.Linear(hidden_dim, dict_size)
        self.to(device)

    def encoder(self, x):
        #Encodeur
        
        # Couche GRU
        #    out: (batch_size, seq_len, n_hidden)
        #    hidden: (n_layers, batch_size, n_hidden)
        out, hidden = self.encoder_rnn(x)

        return out, hidden
    
    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query)

        # 1. Calcul des scores d'attention
        # (batch, 1, n_hidden) @ (batch, n_hidden, seq_len) -> (batch, 1, seq_len)
        scores = torch.bmm(query, values.transpose(1, 2))
        
        # 2. Normalisation des scores avec Softmax
        # Appliqué sur la dimension de la séquence d'entrée (dim=2)
        attention_weights = F.softmax(scores, dim=2)
        
        # 3. Calcul du vecteur de contexte
        # (batch, 1, seq_len) @ (batch, seq_len, n_hidden) -> (batch, 1, n_hidden)
        attention_output = torch.bmm(attention_weights, values)

        return attention_output, attention_weights
    
    def decoder(self, encoder_outs, hidden):
        # Décodeur avec attention

        # Initialisation des variables
        max_len_out = self.max_len
        max_len_in = encoder_outs.size(1) # Longueur de la séquence d'entrée
        batch_size = hidden.shape[1] # Taille de la batch

        # Jeton d'entrée initial (<sos> = 0)
        vec_in = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        
        # Tenseurs pour stocker les résultats
        vec_out = torch.zeros((batch_size, max_len_out, self.dict_size)).to(self.device)
        attention_weights = torch.zeros((batch_size, max_len_out, max_len_in)).to(self.device)

        # Boucle pour tous les symboles de sortie
        for i in range(max_len_out):          
            # 1. Plongement lexical de l'entrée
            # 'embedded' a la forme (batch_size, 1, n_hidden)
            embedded = self.embedding(vec_in)

            # 2. Calcul de l'attention
            # L'état caché 'hidden' (de la couche supérieure) est la "query"
            # 'query' a la forme (batch_size, 1, n_hidden)
            query = hidden[-1].unsqueeze(1) # Prend la dernière couche et ajoute une dimension
            
            # 'context' (batch, 1, n_hidden), 'attn_w' (batch, 1, seq_len)
            context, attn_w = self.attentionModule(query, encoder_outs)
            
            # Stocker les poids d'attention (pour visualisation)
            # squeeze() enlève les dimensions superflues
            attention_weights[:, i, :] = attn_w.squeeze(1)

            # 3. Combiner le mot d'entrée (embedded) et le contexte
            # (batch, 1, n_hidden) + (batch, 1, n_hidden) -> (batch, 1, 2*n_hidden)
            rnn_input = torch.cat((embedded, context), dim=2)
            
            # 4. Passe avant dans la couche GRU du décodeur
            # rnn_out: (batch, 1, n_hidden)
            # hidden: (n_layers, batch, n_hidden)
            rnn_out, hidden = self.decoder_rnn(rnn_input, hidden)
            
            # 5. Couche linéaire de sortie (Logits)
            # output: (batch, 1, dict_size['en'])
            output = self.fc_out(rnn_out.squeeze(1))
            
            # 6. Stocker les logits
            vec_out[:, i, :] = output
            
            # 7. Le mot prédit devient l'entrée pour la prochaine itération
            # output.argmax(1) -> (B)
            # .unsqueeze(1) -> (B, 1)
            vec_in = output.argmax(1).unsqueeze(1)

        return vec_out, hidden, attention_weights

    def forward(self, x):
        # x: (batch_size, seq_len_in, input_dim) - par ex. (B, 150, 2)
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out,h)
        return out, hidden, attn
