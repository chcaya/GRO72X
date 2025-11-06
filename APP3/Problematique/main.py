# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hiver 2021

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = True           # Forcer a utiliser le cpu?
    training = False            # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    n_epochs = 50              # 50 époques est un bon point de départ
    batch_size = 16            # Taille de lot standard
    lr = 0.001                 # Taux d'apprentissage
    hidden_dim = 16           # Dimension de l'état caché
    n_layers = 2               # Nombre de couches pour le GRU
    train_val_split = 0.8      # 80% pour l'entraînement, 20% pour la validation

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset_trainval = HandwrittenWords(filename='data_trainval.p')
    dataset_test = HandwrittenWords(filename='data_test.p')
    
    # Séparation de l'ensemble de données (entraînement et validation)
    n_train_samp = int(len(dataset_trainval) * train_val_split)
    n_val_samp = len(dataset_trainval) - n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_trainval, [n_train_samp, n_val_samp])

    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    dataload_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # Instanciation du model
    # Récupération des propriétés du dataset pour le modèle
    int2symb = dataset_trainval.int2symb
    symb2int = dataset_trainval.symb2int
    dict_size = dataset_trainval.dict_size
    maxlen = dataset_trainval.max_len_target

    model = trajectory2seq(hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen)
    model = torch.compile(model)
    model = model.to(device)

    # Afficher le nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Nombre de poids (paramètres) : {total_params}')

    # Initialisation des variables
    best_val_loss = np.inf
    train_loss_history = []
    val_loss_history = []
    if learning_curves:
        fig, ax = plt.subplots(1)

    if training:

        # Fonction de coût et optimizateur
        # Nous ignorons l'index du padding lors du calcul du coût
        criterion = nn.CrossEntropyLoss(ignore_index=symb2int[dataset_trainval.pad_symbol])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            model.train()
            running_loss_train = 0
            for in_seq, target_seq in dataload_train:
                in_seq = in_seq.to(device)
                target_seq = target_seq.to(device)

                # 1. Remise à zéro des gradients
                optimizer.zero_grad()

                # 2. Passe avant
                output_logits, hidden, attn = model(in_seq) # Forme: (B, L, C)
                
                # 3. Calcul du coût
                # CrossEntropyLoss s'attend à (B, C, L) pour les logits et (B, L) pour la cible
                loss = criterion(output_logits.permute(0, 2, 1), target_seq)

                # 4. Rétropropagation
                loss.backward()

                # 5. Mise à jour des poids
                optimizer.step()
                
                running_loss_train += loss.item()
            
            # Validation
            model.eval()
            running_loss_val = 0
            with torch.no_grad():
                for in_seq, target_seq in dataload_val:
                    in_seq = in_seq.to(device)
                    target_seq = target_seq.to(device)

                    # 1. Passe avant
                    output_logits, hidden, attn = model(in_seq) # (B, L, C)
                    
                    # 2. Calcul du coût
                    loss = criterion(output_logits.permute(0, 2, 1), target_seq)
                    
                    running_loss_val += loss.item()

            # Ajouter les loss aux listes
            epoch_train_loss = running_loss_train / len(dataload_train)
            epoch_val_loss = running_loss_val / len(dataload_val)
            train_loss_history.append(epoch_train_loss)
            val_loss_history.append(epoch_val_loss)
            
            print(f'Epoch {epoch}/{n_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

            # Enregistrer les poids
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model._orig_mod.state_dict(), 'best_model_weights.pt')
                print('   -> Nouveau meilleur modèle sauvegardé!')


            # Affichage
            if learning_curves:
                # visualization
                ax.cla()
                ax.plot(train_loss_history, label='Training Loss')
                ax.plot(val_loss_history, label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Cross-Entropy Loss')
                ax.legend()
                plt.draw()
                plt.pause(0.01)

    if test:
        # Évaluation
        print("\n--- Évaluation sur l'ensemble de test ---")
        # 1. Charger le meilleur modèle sauvegardé
        model = trajectory2seq(hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen)
        model.load_state_dict(torch.load('best_model_weights.pt', map_location=device))
        model.to(device)
        model.eval()
        
        true_list = []
        pred_list = []

        # Charger les données de tests
        with torch.no_grad():
            for in_seq, target_seq in dataload_test:
                in_seq = in_seq.to(device)
                
                # 1. Prédire la séquence de logits
                output_logits, hidden, attn = model(in_seq) # (B, L, C)
                
                # 2. Obtenir les indices des caractères prédits (le plus probable)
                predictions = output_logits.argmax(dim=2) # (B, L)
                
                # 3. Convertir les tenseurs en listes de mots pour les métriques
                for i in range(target_seq.size(0)):
                    # Vrai mot
                    true_tokens = target_seq[i].numpy()
                    true_word_chars = [int2symb[token] for token in true_tokens if token != symb2int[dataset_trainval.pad_symbol]]
                    true_word = ''.join(true_word_chars).split('<eos>')[0] # Arrêter à <eos>
                    
                    # Mot prédit
                    pred_tokens = predictions[i].cpu().numpy()
                    pred_word_chars = [int2symb[token] for token in pred_tokens if token != symb2int[dataset_trainval.pad_symbol]]
                    pred_word = ''.join(pred_word_chars).split('<eos>')[0]
                    
                    true_list.append(true_word)
                    pred_list.append(pred_word)

        # Affichage de l'attention
        # À compléter (si nécessaire)

        # Affichage des résultats de test
        print('\n--- Exemples de résultats de test ---')
        for i in range(min(5, len(true_list))): # Afficher les 5 premiers exemples
            print(f"  Vrai (True):  \t{true_list[i]}")
            print(f"  Prédit (Pred):\t{pred_list[i]}\n")
        
        # Affichage de la matrice de confusion
        print('\n--- Matrice de confusion (par caractère) ---')
        # Concaténer tous les caractères
        true_chars = list("".join(true_list))
        pred_chars = list("".join(pred_list))
        
        # Assurer que les listes ont la même longueur pour la matrice
        min_len = min(len(true_chars), len(pred_chars))
        true_chars = true_chars[:min_len]
        pred_chars = pred_chars[:min_len]
        
        # On ignore les jetons spéciaux
        ignore_tokens = [symb2int['<pad>'], symb2int['<sos>'], symb2int['<eos>']]
        conf_matrix = confusion_matrix(true_chars, pred_chars, ignore=ignore_tokens)
        
        # Afficher la matrice (peut être très grande)
        # print(conf_matrix) 
        
        # Calculer et afficher l'exactitude (Accuracy) par caractère
        correct_chars = np.diag(conf_matrix).sum()
        total_chars = conf_matrix.sum()
        accuracy = correct_chars / total_chars
        print(f'Exactitude (Accuracy) par caractère: {accuracy * 100:.2f}%')

        # Calculer la distance d'édition moyenne
        total_dist = sum(edit_distance(list(t), list(p)) for t, p in zip(true_list, pred_list))
        avg_dist = total_dist / len(true_list)
        print(f'Distance d\'édition moyenne (Levenshtein): {avg_dist:.4f}')

        pass