import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(UNet, self).__init__()
        # ------------------------ Laboratoire 2 - Question 5 - Début de la section à compléter ------------------------
        self.hidden = 32

        # --- Encoder (Downsampling path) ---

        # Down 1: Input -> 32 channels
        self.conv_1_1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu_1_1 = nn.ReLU(inplace=True)
        self.conv_1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu_1_2 = nn.ReLU(inplace=True)

        # Down 2: 32 -> 64 channels
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu_2_1 = nn.ReLU(inplace=True)
        self.conv_2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu_2_2 = nn.ReLU(inplace=True)

        # Down 3: 64 -> 128 channels
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu_3_1 = nn.ReLU(inplace=True)
        self.conv_3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu_3_2 = nn.ReLU(inplace=True)

        # Down 4: 128 -> 256 channels
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu_4_1 = nn.ReLU(inplace=True)
        self.conv_4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu_4_2 = nn.ReLU(inplace=True)

        # Down 5 / Bottleneck: 256 -> 512 channels
        self.maxpool_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_5_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu_5_1 = nn.ReLU(inplace=True)
        self.conv_5_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu_5_2 = nn.ReLU(inplace=True)

        # --- Decoder (Upsampling path) ---

        # Up 5 / Bottleneck: 512 -> 256 channels
        self.conv_5_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu_5_2 = nn.ReLU(inplace=True)

        # Up 6: 256 -> 128 channels
        self.upsample_6 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv_6_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # 256 from upsample + 256 from skip
        self.relu_6_1 = nn.ReLU(inplace=True)
        self.conv_6_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu_6_2 = nn.ReLU(inplace=True)

        # Up 7: 128 -> 64 channels
        self.upsample_7 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv_7_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # 128 from upsample + 128 from skip
        self.relu_7_1 = nn.ReLU(inplace=True)
        self.conv_7_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu_7_2 = nn.ReLU(inplace=True)

        # Up 8: 64 -> 32 channels
        self.upsample_8 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_8_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # 64 from upsample + 64 from skip
        self.relu_8_1 = nn.ReLU(inplace=True)
        self.conv_8_2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu_8_2 = nn.ReLU(inplace=True)

        # Up 9: 32 -> 32 channels
        self.upsample_9 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv_9_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1) # 32 from upsample + 32 from skip
        self.relu_9_1 = nn.ReLU(inplace=True)
        self.conv_9_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu_9_2 = nn.ReLU(inplace=True)

        # Final 1x1 convolution to map to number of classes
        self.output_conv = nn.Conv2d(self.hidden, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        # Down 1
        x = self.relu_1_1(self.conv_1_1(x))
        x1 = self.relu_1_2(self.conv_1_2(x))

        # Down 2
        x = self.maxpool_2(x1)
        x = self.relu_2_1(self.conv_2_1(x))
        x2 = self.relu_2_2(self.conv_2_2(x))

        # Down 3
        x = self.maxpool_3(x2)
        x = self.relu_3_1(self.conv_3_1(x))
        x3 = self.relu_3_2(self.conv_3_2(x))

        # Down 4
        x = self.maxpool_4(x3)
        x = self.relu_4_1(self.conv_4_1(x))
        x4 = self.relu_4_2(self.conv_4_2(x))

        # Down 5 (Bottleneck)
        x = self.maxpool_5(x4)
        x = self.relu_5_1(self.conv_5_1(x))

        # --- Decoder ---
        
        # Up 5 (Bottleneck)
        x = self.relu_5_2(self.conv_5_2(x))

        # Up 6
        x = self.upsample_6(x)
        x = torch.cat([x, x4], dim=1)
        x = self.relu_6_1(self.conv_6_1(x))
        x = self.relu_6_2(self.conv_6_2(x))

        # Up 7
        x = self.upsample_7(x)
        x = torch.cat([x, x3], dim=1)
        x = self.relu_7_1(self.conv_7_1(x))
        x = self.relu_7_2(self.conv_7_2(x))

        # Up 8
        x = self.upsample_8(x)
        x = torch.cat([x, x2], dim=1)
        x = self.relu_8_1(self.conv_8_1(x))
        x = self.relu_8_2(self.conv_8_2(x))

        # Up 9
        x = self.upsample_9(x)
        x = torch.cat([x, x1], dim=1)
        x = self.relu_9_1(self.conv_9_1(x))
        x = self.relu_9_2(self.conv_9_2(x))
        
        # Output
        out = self.output_conv(x)

        return out
        # ------------------------ Laboratoire 2 - Question 5 - Fin de la section à compléter --------------------------
