# GRO722 Laboratoire 1
# Auteurs: Jean-Samuel Lauzon et Jonathan Vincent
# Hiver 2021
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle

class SignauxDataset(Dataset):
    """Ensemble de signaux continus à differentes fréquences"""
    def __init__(self, filename='data.p'):
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

    def __len__(self):

        # ------------------------ Laboratoire 1 - Question 1 - Début de la section à compléter ----------------------------
        
        return len(self.data)

        # ---------------------- Laboratoire 1 - Question 1 - Fin de la section à compléter --------------------------------

    def __getitem__(self, idx):

        # ------------------------ Laboratoire 1 - Question 1 - Début de la section à compléter ----------------------------
        
        # 1. Récupérer l'échantillon (qui est un tuple) à l'index donné
        sample = self.data[idx]
        
        # 2. Séparer l'entrée et la cible
        input_sequence = sample[0]
        target_sequence = sample[1]

        # 3. Convertir les séquences en tenseurs Pytorch de type float
        # (requis pour les données continues)
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32)
        target_tensor = torch.tensor(target_sequence, dtype=torch.float32)
        
        return input_tensor, target_tensor

        # ---------------------- Laboratoire 1 - Question 1 - Fin de la section à compléter --------------------------------
    
    def visualize(self, idx):
        input_sequence, target_sequence = [i.numpy() for i in self[idx]]
        t = range(len(input_sequence)+1)
        plt.plot(t[:-1],input_sequence, label='input sequence')
        plt.plot(t[1:],target_sequence,label='target sequence')
        plt.title('Visualization of sample: '+str(idx))
        plt.legend()
        plt.show()

        pass

if __name__=='__main__':
    a = SignauxDataset()
    dataload_train = DataLoader(a, batch_size=2, shuffle=True)
    a.visualize(0)
