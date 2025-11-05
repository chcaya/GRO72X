import torch.nn as nn

class ConvBNAct(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU."""

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k,
                      stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ClassificationNet(nn.Module):
    """
    Simple CNN for multi-label classification over 3 classes.
    - Input:  (N, 1, H, W)  (grayscale)
    - Output: (N, 3) logits (use BCEWithLogitsLoss)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3, dropout_p: float = 0.2):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 32, k=3, s=1, p=1),
            ConvBNAct(32, 32, k=3, s=1, p=1),
            nn.MaxPool2d(2),                    # /2

            ConvBNAct(32, 64, k=3, s=1, p=1),
            ConvBNAct(64, 64, k=3, s=1, p=1),
            nn.MaxPool2d(2),                    # /4

            ConvBNAct(64, 128, k=3, s=1, p=1),
        )

        # (N, 128, 53/4, 53/4) -> (N, 128, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, num_classes, bias=True)  # logits
        )

class ClassificationNetwork(nn.Module):
    """
    A simple CNN for multi-label classification.
    """

    def __init__(self):
        super().__init__()
        # Input shape: (N, 1, 53, 53)
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # Shape: (N, 32, 53, 53)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),
            # Shape: (N, 32, 26, 26)  (53 / 2 = 26, floored)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # Shape: (N, 64, 26, 26)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),
            # Shape: (N, 64, 13, 13)  (26 / 2 = 13)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # Shape: (N, 128, 13, 13)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)

            # Output shape: (N, 128, 13, 13)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Output shape: (N, 128, 1, 1)

        # Input shape: (N, 128, 1, 1)
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            # Shape: (N, 128)

            nn.Dropout(p=0.2),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            # Shape: (N, 64)

            nn.Dropout(p=0.2),
            nn.Linear(64, 3, bias=True)
            
            # Final output shape: (N, 3)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = self.classifier_head(x)
        return x
