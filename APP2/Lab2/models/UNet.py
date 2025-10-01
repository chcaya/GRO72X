import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(UNet, self).__init__()
        # ------------------------ Laboratoire 2 - Question 5 - Début de la section à compléter ------------------------
        self.hidden = None

        # Down 1
        self.conv_1_1 = None
        self.relu_1_1 = None
        self.conv_1_2 = None
        self.relu_1_2 = None

        # Down 2
        self.maxpool_2 = None
        self.conv_2_1 = None
        self.relu_2_1 = None
        self.conv_2_2 = None
        self.relu_2_2 = None

        # Down 3
        self.maxpool_3 = None
        self.conv_3_1 = None
        self.relu_3_1 = None
        self.conv_3_2 = None
        self.relu_3_2 = None

        # Down 4
        self.maxpool_4 = None
        self.conv_4_1 = None
        self.relu_4_1 = None
        self.conv_4_2 = None
        self.relu_4_2 = None

        # Down 5
        self.maxpool_5 = None
        self.conv_5_1 = None
        self.relu_5_1 = None
        self.conv_5_2 = None
        self.relu_5_2 = None

        # Up 6
        self.upsample_6 = None
        self.conv_6_1 = None
        self.relu_6_1 = None
        self.conv_6_2 = None
        self.relu_6_2 = None

        # Up 7
        self.upsample_7 = None
        self.conv_7_1 = None
        self.relu_7_1 = None
        self.conv_7_2 = None
        self.relu_7_2 = None

        # Up 8
        self.upsample_8 = None
        self.conv_8_1 = None
        self.relu_8_1 = None
        self.conv_8_2 = None
        self.relu_8_2 = None

        # Up 9
        self.upsample_9 = None
        self.conv_9_1 = None
        self.relu_9_1 = None
        self.conv_9_2 = None
        self.relu_9_2 = None

        self.output_conv = nn.Conv2d(self.hidden, n_classes, kernel_size=1)

    def forward(self, x):
        # Down 1
        # To do

        # Down 2
        # To do

        # Down 3
        # To do

        # Down 4
        # To do

        # Down 5
        # To do

        # Up 6
        # To do

        # Up 7
        # To do

        # Up 8
        # To do

        # Up 9
        # To do

        # Out
        out = None

        return out
        # ------------------------ Laboratoire 2 - Question 5 - Fin de la section à compléter --------------------------
