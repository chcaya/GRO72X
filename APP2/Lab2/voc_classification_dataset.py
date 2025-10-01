import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VOCSegmentation, VOCDetection


class VOCClassificationDataset(VOCDetection):
    def __init__(self, root, image_set='train', download=True, img_shape=100):
        super().__init__(root=root, image_set=image_set, download=download)
        self.img_shape = img_shape
        self.nb_classe = 21  # Nombre de classes dans le PASCAL VOC
        self.VOC_ID_2_CLASSES = {
            0: 'background',
            1: 'aeroplane',
            2: 'bicycle',
            3: 'bird',
            4: 'boat',
            5: 'bottle',
            6: 'bus',
            7: 'car',
            8: 'cat',
            9: 'chair',
            10: 'cow',
            11: 'diningtable',
            12: 'dog',
            13: 'horse',
            14: 'motorbike',
            15: 'person',
            16: 'pottedplant',
            17: 'sheep',
            18: 'sofa',
            19: 'train',
            20: 'tvmonitor'
        }
        self.VOC_CLASSES_2_ID = {v: k for k, v in self.VOC_ID_2_CLASSES.items()}

    def __getitem__(self, index):
        # Chargement des images et métadonnées provenant de PASCAL VOC
        image = Image.open(self.images[index])
        target_metadata = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())

        # Creation d'un vecteur multi-hot pour la classification
        multi_hot = torch.zeros(self.nb_classe, dtype=torch.float)

        # ------------------------ Laboratoire 2 - Question 1 - Début de la section à compléter ------------------------


        # ------------------------ Laboratoire 2 - Question 1 - Fin de la section à compléter --------------------------


        # Création des transformations
        heigth, width, channel = np.array(image).shape
        transform = transforms.Compose(
            [transforms.CenterCrop(min([width, heigth])),
             transforms.Resize(size=(self.img_shape, self.img_shape))
             ])

        # Application de la transformation de l'image
        image = transform(image)

        # Mettre les données en tenseurs et dans le bon format
        image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255
        target = torch.from_numpy(np.array(multi_hot))

        return image, target


if __name__ == '__main__':
    # Génération des "path"
    dir_path = os.path.dirname(__file__)
    data_path = os.path.join(dir_path, 'data')

    # Chargement du dataset
    dataset_train = VOCClassificationDataset(data_path, image_set='train', download=True, img_shape=500)

    # Affichage
    fig, axs = plt.subplots(3, 1)
    for i in range(3):
        image, target = dataset_train[np.random.randint(0, len(dataset_train))]
        labels = [dataset_train.VOC_ID_2_CLASSES[i] for i, val in enumerate(target) if val == 1]
        axs[i].imshow(image.permute(1, 2, 0))
        axs[i].set_title(', '.join(labels))
        axs[i].get_xaxis().set_ticks([])
        axs[i].get_yaxis().set_ticks([])
    plt.show()
