# GRO722 Laboratoire 1
# Auteurs: Jean-Samuel Lauzon et Jonathan Vincent
# Hiver 2021
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, n_hidden, n_layers=1):
        super(Model, self).__init__()

        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------

        # Stockage des hyperparamètres pour la fonction forward
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        # Définition du réseau récurrent (Elman)
        # input_size = 1 (une seule dimension par pas de temps)
        # hidden_size = n_hidden (nombre de neurones dans l'état caché)
        # batch_first = True (le tenseur d'entrée aura la forme [lot, séquence, features])
        self.rnn = nn.LSTM(
            input_size=1, 
            hidden_size=n_hidden, 
            num_layers=n_layers, 
            batch_first=True
        )
        
        # Couche de sortie linéaire pour mapper l'état caché à la sortie désirée (taille 1)
        self.fc = nn.Linear(in_features=n_hidden, out_features=1)

        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------
    
    def forward(self, x, h=None):

        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------

        # 1. Gestion de l'état caché initial (h)
        # Si h est None, nous l'initialisons à un tenseur de zéros.
        if h is None:
            # Récupération de la taille du lot (batch_size) depuis x
            batch_size = x.size(0)
            
            # Initialisation de h_0 : (nombre_de_couches, taille_du_lot, taille_cachée)
            h_0 = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(x.device)
            # Initialisation de c_0 (cell state)
            c_0 = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(x.device)
            
            # h est maintenant un tuple
            h = (h_0, c_0)

        # 2. Passe avant dans le RNN
        # rnn_out: sortie de la couche récurrente pour chaque pas de temps (batch_size, seq_len, hidden_size)
        # h_next: état caché final (num_layers, batch_size, hidden_size)
        rnn_out, h_next = self.rnn(x, h)
        
        # 3. Passe avant dans la couche linéaire de sortie
        # Nous appliquons la couche linéaire à chaque pas de temps de la sortie du RNN.
        # La forme de 'output' sera (batch_size, seq_len, 1)
        output = self.fc(rnn_out)

        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------

        return output, h_next

if __name__ == '__main__':
    x = torch.zeros((100,2,1)).float()
    model = Model(25)
    print(model(x))
