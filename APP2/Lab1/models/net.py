import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        # self.fc1 = nn.Linear(28 * 28, 10)
        
        # We replace the Linear layer with a Conv2d layer.
        # To make it equivalent, the kernel_size must match the input's spatial dimensions (28x28).
        # in_channels=1 because the input images are grayscale.
        # out_channels=10 for the 10 digit classes.
        # self.fc1_conv = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=28)

        # Block 1: Conv -> BatchNorm -> ReLU -> MaxPool
        # Layer 1: Convolution (4 noyaux de 3x3, remplissage de 1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        # Layer 2: Normalisation de lot (sur 4 canaux)
        self.bn1 = nn.BatchNorm2d(num_features=4)
        # Layer 4: Mise en commun maximale (noyau de 2x2, foulée de 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: Conv -> BatchNorm -> ReLU -> MaxPool
        # Layer 5: Convolution (2 noyaux de 3x3, remplissage de 1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1)
        # Layer 6: Normalisation de lot (sur 2 canaux)
        self.bn2 = nn.BatchNorm2d(num_features=2)

        # Layer 9: Couche pleinement connectée
        # To determine the input size, we trace the dimensions:
        # Input: 28x28 -> After conv1 (padding=1): 28x28 -> After pool1: 14x14
        # -> After conv2 (padding=1): 14x14 -> After pool2: 7x7
        # The final feature map has 2 channels and is 7x7.
        # So, the input features are 2 * 7 * 7 = 98.
        self.fc1 = nn.Linear(in_features=98, out_features=10) # 10 classes for output

        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------

    def forward(self, x):
        # ---------------------- Laboratoire 1 - Question 5 et 6 - Début de la section à compléter ---------------------
        # output = self.fc1(x.view(x.shape[0], -1))

        # 1. Pass the original 2D image (e.g., shape [64, 1, 28, 28]) through the convolutional layer.
        #    We do NOT flatten the input first.
        # x = self.fc1_conv(x) # Output shape will be [64, 10, 1, 1]
        # 2. Flatten the output to match the shape expected by the loss function (e.g., [64, 10]).
        # output = torch.flatten(x, 1)

        # Pass through Block 1
        # Layer 1, 2, 3, 4
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)

        # Pass through Block 2
        # Layer 5, 6, 7, 8
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)

        # Flatten the feature map for the fully connected layer
        # The shape is now (batch_size, 2, 7, 7), we flatten it to (batch_size, 98)
        x = torch.flatten(x, 1)

        # Pass through the final fully connected layer
        # Layer 9
        output = self.fc1(x)

        # ---------------------- Laboratoire 1 - Question 5 et 6 - Fin de la section à compléter -----------------------
        return output
