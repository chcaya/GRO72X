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

        self.pooling_and_fc = nn.Sequential(
            # This layer averages each of the 32 channels down to a 1x1 size
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # The input to the linear layer is now just the number of channels (32)
            nn.Linear(32, 3) 
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pooling_and_fc(x)
        return x
