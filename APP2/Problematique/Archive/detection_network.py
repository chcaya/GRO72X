import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    The core building block of a ResNet. It contains a "shortcut" or "skip connection"
    that adds the input of the block to its output.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut path to handle changes in dimensions or channels
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # The residual connection: add the original input (shortcut) to the processed output
        out = self.main_path(x) + self.shortcut(x)
        out = self.final_relu(out)
        return out

class DetectionNetwork(nn.Module):
    """
    A ResNet-style network for detection, with a total of ~321k parameters.
    """
    def __init__(self):
        super().__init__()
        
        # Initial "stem" layer
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) # Image size -> 26x26
        )
        
        # Stack of residual blocks
        self.layer1 = ResidualBlock(16, 16, stride=1)
        self.layer2 = ResidualBlock(16, 32, stride=2) # Downsample -> 13x13
        self.layer3 = ResidualBlock(32, 64, stride=2) # Downsample -> 6x6 (floor)
        
        # The same head as before, but with a smaller intermediate layer
        # to keep the parameter count low.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 64), 
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 3 * 7)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.head(x)
        return x.view(-1, 3, 7)
