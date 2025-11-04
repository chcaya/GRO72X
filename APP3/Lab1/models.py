# GRO722 Laboratoire 1
# Auteurs: Jean-Samuel Lauzon et Jonathan Vincent
# Hiver 2021
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, n_hidden, n_layers=1):
        super(Model, self).__init__()

        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------


        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------
    
    def forward(self, x, h=None):

        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------


        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------

        return x, h

if __name__ == '__main__':
    x = torch.zeros((100,2,1)).float()
    model = Model(25)
    print(model(x))