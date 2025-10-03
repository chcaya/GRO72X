import torch.nn as nn

class ClassificationNetwork(nn.Module):
    """
    A simple CNN for multi-label classification.
    """

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 53 -> 26
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 26 -> 13
        )

        self.classifier_head = nn.Sequential(
            # This layer takes the 13x13 feature map and resizes it to 4x4
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            # The input is now much smaller: 32 channels * 4 * 4 features
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier_head(x)
        return x
