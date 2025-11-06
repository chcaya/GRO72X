import torch
import torch.nn as nn
import torch.nn.functional as F

    
class DoubleConv(nn.Module):
    """Helper module for U-Net: (convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DetectionNetwork(nn.Module):
    """
    A ResNet-style network for detection, with a total of ~321k parameters.
    """
    def __init__(self):
        super().__init__()

        # Core layers
        self.layer1 = nn.Sequential(
            DoubleConv(1, 16),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2) # Downsample -> 26x26
        )
        self.layer2 = nn.Sequential(
            DoubleConv(16, 32),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2) # Downsample -> 13x13
        )
        self.layer3 = nn.Sequential(
            DoubleConv(32, 64),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2) # Downsample -> 6x6
        )

        # Output layers
        self.neck = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(p=0.2),
            nn.Linear(64 * 6 * 6, 64), 
            nn.LeakyReLU(inplace=True),

            nn.Dropout(p=0.2)
        )

        self.obj_head = nn.Linear(64, 3 * 1)               # (N,64) -> (N,3)
        self.box_head = nn.Linear(64, 3 * 3)               # (N,64) -> (N,9)
        self.cls_head = nn.Linear(64, 3 * 3)               # (N,64) -> (N,9)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.neck(x)

        obj = self.obj_head(x).view(-1, 3, 1)
        box = self.box_head(x).view(-1, 3, 3)
        cls = self.cls_head(x).view(-1, 3, 3)

        return torch.cat([obj, box, cls], dim=-1)  # (N, num_anchors, 1+3+num_classes)


class DetectionLoss(nn.Module):
    """
    Perte composite simple pour nos cibles (N, A, 5):
      [:, :, 0] presence {0/1}
      [:, :, 1:4] x, y, size  in [0,1]
      [:, :, 4]   class_idx   in {0..C-1}
    """

    def __init__(self, lambda_obj=1.0, lambda_noobj=0.5, lambda_bbox=40.0, lambda_class=2.0):
        super().__init__()
        # --- Hyperparameters for balancing the loss components ---
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_bbox = lambda_bbox
        self.lambda_class = lambda_class

    def forward(self, prediction, target):
        """
        A standard object detection loss function that combines confidence, bounding
        box, and classification losses.

        :param prediction: The (N, 3, 7) output from the model.
                            Format: [conf, x, y, size, score_c0, score_c1, score_c2]
        :param target: The (N, 3, 5) ground truth tensor.
                        Format: [objectness, x, y, size, class_index]
        """        
        # --- Create a mask to find which prediction slots contain an object ---
        obj_mask = target[..., 0] == 1
        noobj_mask = target[..., 0] == 0

        # --- 1. Confidence (Objectness) Loss ---
        loss_conf_obj = F.binary_cross_entropy_with_logits(
            prediction[..., 0][obj_mask],
            target[..., 0][obj_mask]
        )
        loss_conf_noobj = F.binary_cross_entropy_with_logits(
            prediction[..., 0][noobj_mask],
            target[..., 0][noobj_mask]
        )
        loss_confidence = (self.lambda_obj * loss_conf_obj) + (self.lambda_noobj * loss_conf_noobj)
        
        # --- 2. Bounding Box Loss (Localization) ---
        loss_bbox = torch.tensor(0.0, device=prediction.device)
        if obj_mask.sum() > 0:
            bbox_pred_logits = prediction[..., 1:4][obj_mask]
            bbox_pred = torch.sigmoid(bbox_pred_logits)
            bbox_target = target[..., 1:4][obj_mask]
            loss_bbox = F.smooth_l1_loss(bbox_pred, bbox_target, reduction='mean')
            
        # --- 3. Classification Loss ---
        loss_class = torch.tensor(0.0, device=prediction.device)
        if obj_mask.sum() > 0:
            class_pred_logits = prediction[..., 4:][obj_mask]
            target_class_indices = target[..., 4][obj_mask].long()
            loss_class = F.cross_entropy(class_pred_logits, target_class_indices, reduction='mean')

        # --- Final Combined Loss ---
        print(f"Conf obj Loss: {(self.lambda_obj*loss_conf_obj).item():.4f}, \
                Conf no obj Loss: {(self.lambda_noobj * loss_conf_noobj).item():.4f}, \
                BBox Loss: {(self.lambda_bbox * loss_bbox).item():.4f}, \
                Class Loss: {(self.lambda_class * loss_class).item():.4f}")
        total_loss = loss_confidence + (self.lambda_bbox * loss_bbox) + (self.lambda_class * loss_class)
        
        return total_loss
