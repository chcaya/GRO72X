import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Helper module for U-Net: (convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SegmentationNetwork(nn.Module):
    """
    A lightweight U-Net architecture for semantic segmentation.
    """

    def __init__(self, n_channels=1, n_classes=4):
        super().__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)  # 64 (skip) + 64 (upsample)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, output_padding=1)
        self.conv2 = DoubleConv(64, 32)  # 32 (skip) + 32 (upsample)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)
        logits = self.outc(x)
        return logits
