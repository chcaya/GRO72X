import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import cm
from matplotlib.colors import ListedColormap
from torchvision import transforms

# Inclure le modèle
from models.UNet import UNet

# Génération des "path"
dir_path = os.path.dirname(__file__)
test_data_path = os.path.join(dir_path, 'test_images')
weights_path = os.path.join(dir_path, 'weights', 'unet_coco.pt')
test_images = [os.path.basename(x) for x in glob.glob(test_data_path + '/*')]

class_names = ['background', 'person', 'dog']

# ---------------- Paramètres ----------------#
use_cpu = True  # Forcer à utiliser le cpu?
input_channels = 3  # Nombre de canaux d'entrée
nb_classes = 3  # Nombre de classes
img_shape = 256
# ------------ Fin des paramètres ------------#

if __name__ == '__main__':
    # Init affichage
    fig, axs = plt.subplots(2)

    # New colormap
    newcolors = cm.tab10(np.linspace(0, 1, 10))
    white = np.array([1, 1, 1, 1])
    green = np.array([57 / 256, 178 / 256, 85 / 256, 1])
    blue = np.array([91 / 256, 101 / 256, 195 / 256, 1])
    newcolors = [white, newcolors[0], newcolors[2]]
    newcmp = ListedColormap(newcolors)

    # Choix du device
    use_cuda = not use_cpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Instancier le modèle sur CPU
    model = UNet(input_channels, nb_classes).to(device)
    state_dict = torch.load(weights_path, weights_only=False, map_location=lambda storage, loc: storage).state_dict()
    model.load_state_dict(state_dict)
    model.eval()

    # Affiche le résume du modèle
    print('Model : \n', model, '\n')

    # Affichage d'une prediction et des courbes d'apprentissages
    for idx in range(len(test_images)):
        # Chargement de l'image
        img = Image.open(os.path.join(test_data_path, test_images[idx]))
        height, width, channel = np.array(img).shape
        transform = transforms.Compose([transforms.CenterCrop(min([width, height])),
                                        transforms.Resize(size=(img_shape, img_shape))])
        image = transform(img)
        image = np.array(image)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        image = image / 255

        output = model(image[None, :, :, :].to(device))
        output_t = torch.argmax(output[0], dim=0).detach().cpu().numpy()

        axs[0].cla()
        axs[1].cla()

        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].set_title('Input image')
        axs[1].imshow(output_t, vmin=0, vmax=2, cmap=newcmp)
        axs[0].get_xaxis().set_ticks([])
        axs[0].get_yaxis().set_ticks([])
        axs[1].get_xaxis().set_ticks([])
        axs[1].get_yaxis().set_ticks([])
        labels = []
        for i, val in enumerate(class_names):
            if i in output_t and i:
                labels.append(class_names[i])
        axs[1].set_title(', '.join(labels))

        fig.show()
        plt.pause(0.001)
        ans = input('Press Enter to display the next prediction ...')
