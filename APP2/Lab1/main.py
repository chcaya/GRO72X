import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

# Inclure le modèle
from models.net import Net

if __name__ == '__main__':
    # Génération des 'path'
    dir_path = os.path.dirname(__file__)
    data_path = os.path.join(dir_path, 'data')
    digits_path = os.path.join(dir_path, 'digits')
    weights_path = os.path.join(dir_path, 'weights', 'mnist_best.pt')

    # ---------------- Paramètres et hyperparamètres ----------------#
    train = False  # Entraînement?
    test = False  # Tester avec le meilleur modèle?
    use_cpu = True  # Forcer a utiliser le cpu?
    save_model = True  # Sauvegarder le meilleur modèle ?

    batch_size = 64  # Taille des lots pour l'entraînement
    val_test_batch_size = 64  # Taille des lots pour validation et test
    epochs = 10  # Nombre d'itérations (epochs)
    train_val_split = 0.7  # Proportion d'échantillons
    lr = 0.001  # Pas d'apprentissage
    random_seed = 1  # Pour répétabilité
    num_workers = 6  # Nombre de threads pour chargement des données
    # ------------ Fin des paramètres et hyper-parametres ------------#

    # Initialisation des objets et variables
    best_val = np.inf
    torch.manual_seed(random_seed)
    np.random.seed(seed=random_seed)

    # Choix du device
    use_cuda = not use_cpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Instancier le modèle
    model = Net().to(device)

    # Imprime le resume du model
    print('Model : \n', model, '\n')



    # ------------------------ Laboratoire 1 - Question 1 - Début de la section à compléter ----------------------------




    # ---------------------- Laboratoire 1 - Question 1 - Fin de la section à compléter --------------------------------




    # ------------------------ Laboratoire 1 - Question 2 - Début de la section à compléter ----------------------------
    # Creation des dataloaders
    train_loader = [None]
    val_loader = [None]
    test_loader = [None]
    # ---------------------- Laboratoire 1 - Question 2 - Fin de la section à compléter --------------------------------


    # ---------------------- Laboratoire 1 - Question 3 - Début de la section à compléter ------------------
    # Création de l'optimisateur et de la fonction de coût
    optimizer = None
    loss_criterion = None
    # ---------------------- Laboratoire 1 - Question 3 - Fin de la section à compléter --------------------


    if train:
        print('Starting training')
        epochs_train_losses = []  # Historique des coûts
        epochs_val_losses = []  # Historique des coûts

        for epoch in range(1, epochs + 1):
            # Entraînement
            model.train()
            running_loss = 0

            # Boucle pour chaque lot
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)



                # ---------------------- Laboratoire 1 - Question 3 - Début de la section à compléter ------------------

                # ---------------------- Laboratoire 1 - Question 3 - Fin de la section à compléter --------------------



                # Affichage pendant l'entraînement
                if batch_idx % 10 == 0:
                    print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f}'.format(
                        epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                                       100. * batch_idx / len(train_loader), running_loss / (batch_idx + 1)), end='\r')

            # Historique des coûts
            epochs_train_losses.append(running_loss / len(train_loader))

            # Validation
            model.eval()
            val_loss = 0
            accuracy = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)



                    # ---------------------- Laboratoire 1 - Question 4 - Début de la section à compléter --------------

                # ---------------------- Laboratoire 1 - Question 4 - Fin de la section à compléter --------------------



            # Historique des coûts de validation
            val_loss /= len(val_loader)
            epochs_val_losses.append(val_loss)
            print('\nValidation - Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                val_loss, 100. * accuracy))

            # Sauvegarde du meilleur modèle
            if val_loss < best_val and save_model:
                best_val = val_loss
                print('Saving new best model\n')
                torch.save(model, weights_path)

            # Affichage des courbes d'apprentissages
            plt.clf()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.plot(range(1, epoch + 1), epochs_train_losses, color='blue', label='Training loss', linestyle=':')
            plt.plot(range(1, epoch + 1), epochs_val_losses, color='red', label='Validation loss', linestyle='-.')
            plt.legend()
            plt.draw()
            plt.pause(0.001)

        plt.show()

    if test:
        print('Starting test')
        # Chargement du meilleur modèle
        model = torch.load(weights_path, weights_only=False)

        model.eval()
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)


                # ---------------------- Laboratoire 1 - Question 4 - Début de la section à compléter ------------------

                output = None

            # ---------------------- Laboratoire 1 - Question 4 - Fin de la section à compléter ------------------------



        test_loss /= len(test_loader)
        print('Test - Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * accuracy))

        continue_pred = True
        data = data.cpu().numpy()
        while continue_pred:

            index = np.random.randint(0, data.shape[0])
            img = data[index]
            number_pred = torch.argmax(output, dim=1)[index].cpu().numpy()
            plt.clf()
            plt.title('Prediction : ' + str(number_pred))
            plt.imshow(img[0], cmap='gray')
            plt.draw()
            plt.pause(0.01)

            ans = input('Do you want to display an other test? (y/n):')
            if ans == 'n':
                continue_pred = False
