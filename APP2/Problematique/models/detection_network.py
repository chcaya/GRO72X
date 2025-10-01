import torch.nn as nn

# class DetectionNetwork(nn.Module):
#     """
#     A CNN with a regression head for object detection.
#     """

#     def __init__(self):
#         super().__init__()
#         self.backbone = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 53x53
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # 26x26
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # 13x13
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)  # 6x6 (floor)
#         )
#         self.head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 6 * 6, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3 * 5)  # 3 objects, 5 values each (obj, x, y, w, h)
#         )

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.head(x)
#         # Reshape to (N, 3, 5) to match target format
#         return x.view(-1, 3, 5)
    
class DetectionNetwork(nn.Module):
    """
    A CNN with a strong backbone and a lightweight, direct regression head.
    """
    def __init__(self):
        super().__init__()
        # The backbone remains the same, producing a rich 6x6 feature map.
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output is (N, 64, 6, 6)
        )

        # The NEW head is extremely simple and direct.
        self.head = nn.Sequential(
            nn.Flatten(),
            # Directly map the 2304 features to the 15 output values.
            nn.Linear(64 * 6 * 6, 3 * 5) 
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        # Reshape to the required (N, 3, 5) target format.
        return x.view(-1, 3, 5)
