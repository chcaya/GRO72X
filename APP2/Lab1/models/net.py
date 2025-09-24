import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        self.fc1 = nn.Linear(28 * 28, 10)

        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------

    def forward(self, x):
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        output = self.fc1(x.view(x.shape[0], -1))
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------
        return output
