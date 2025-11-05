import json
import os

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MAX_SHAPE_COUNT = 3
SHAPE_TO_CLASS = {
    'circle': 0,
    'triangle': 1,
    'cross': 2
}


class ConveyorSimulator(Dataset):
    def __init__(self, data_path, transform=None, num_classes=4):
        self._transform = transform
        self._data_path = data_path
        self._json_path = os.path.join(data_path, 'metaData.json')
        self._num_classes = num_classes
        with open(self._json_path) as f:
            self._metadata = json.load(f)

    def __len__(self):
        return len(self._metadata.keys())

    def __getitem__(self, index):
        img_id = list(self._metadata.keys())[index]
        image = Image.open(os.path.join(self._data_path, 'images', img_id))
        mask_image = np.array(Image.open(os.path.join(self._data_path, 'masks', img_id)))

        # 1. Extract the single channel that contains the class indices.
        # Based on your original code, this is the blue channel (index 2).
        # This is now a 2D NumPy array of shape (H, W) where each pixel is a class ID.
        segmentation_target_np = mask_image[:, :, 2]

        # --- 1. Create empty lists to hold valid data ---
        boxes_to_transform = []
        class_labels_to_transform = []

        # --- 2. Loop through source data and perform validity check ---
        for i in range(len(self._metadata[img_id]['shape'])):
            size = self._metadata[img_id]['size'][i]
            pos_x, pos_y = self._metadata[img_id]['position'][i]
            class_id = SHAPE_TO_CLASS[self._metadata[img_id]['shape'][i]]
            
            # Assuming width = height = size for the check
            cx, cy, w, h = pos_x, pos_y, size, size

            # 1. Convert from [cx, cy, w, h] to [x_min, y_min, x_max, y_max]
            x_min = cx - w / 2
            y_min = cy - h / 2
            x_max = cx + w / 2
            y_max = cy + h / 2

            # 2. Clip the coordinates to the valid [0.0, 1.0] range
            x_min = np.clip(x_min, 0, 1)
            y_min = np.clip(y_min, 0, 1)
            x_max = np.clip(x_max, 0, 1)
            y_max = np.clip(y_max, 0, 1)

            # 3. Convert back to the 'yolo' format [cx, cy, w, h] that albumentations expects
            clipped_w = x_max - x_min
            clipped_h = y_max - y_min
            clipped_cx = x_min + clipped_w / 2
            clipped_cy = y_min + clipped_h / 2
            
            # Add the new, valid box to the list for transformation
            # Only add if the box still has a positive area after clipping
            if clipped_w > 0 and clipped_h > 0:
                boxes_to_transform.append([clipped_cx, clipped_cy, clipped_w, clipped_h])
                class_labels_to_transform.append(class_id)

        # Process segmentation target (this part is fine)
        segmentation_target_np = np.array(mask_image)[:, :, 2]
        segmentation_target_np[segmentation_target_np == 0] = 4
        segmentation_target_np -= 1
        
        # --- Apply Synchronized Transform (if it exists) ---
        image_np = np.array(image)
        if self._transform:
            transformed = self._transform(
                image=image_np, 
                masks=[segmentation_target_np],
                bboxes=boxes_to_transform, 
                class_ids=class_labels_to_transform
            )
            image = transformed['image']
            segmentation_target = transformed['masks'][0]
            transformed_bboxes = transformed['bboxes']
            transformed_class_ids = transformed['class_ids']
        else:
            # If no transform, just use the original valid data
            image = transforms.ToTensor()(image)
            segmentation_target = segmentation_target_np
            transformed_bboxes = boxes_to_transform
            transformed_class_ids = class_labels_to_transform

        # --- 4. After the loop, create the final padded tensors ---
        boxes_tensor = torch.zeros((MAX_SHAPE_COUNT, 5))
        for i in range(len(transformed_bboxes)):
            cx, cy, w, h = transformed_bboxes[i]
            class_id = transformed_class_ids[i]
            boxes_tensor[i] = torch.tensor([1.0, cx, cy, (w+h)/2, class_id])

        class_labels_tensor = torch.zeros((3))
        for cid in transformed_class_ids:
            class_labels_tensor[int(cid)] = 1.0

        # print('***boxes_to_transform***')
        # print(boxes_to_transform)

        # print('***transformed_bboxes***')
        # print(transformed_bboxes)

        # print('***boxes_tensor***')
        # print(boxes_tensor)

        return image[0:1, :, :], torch.as_tensor(segmentation_target, dtype=torch.long), boxes_tensor, class_labels_tensor


if __name__ == '__main__':
    dir_path = os.path.dirname(__file__)
    training_path = os.path.join(dir_path, 'data', 'training')

    t = transforms.Compose([transforms.ToTensor()])

    train_set = ConveyorSimulator(training_path, t)

    train_loader = DataLoader(train_set, batch_size=500, shuffle=True, num_workers=6)

    circle = 0
    triangle = 0
    cross = 0

    for image, masks, boxes, labels in train_loader:
        for i in range(labels.shape[0]):
            if labels[i][0] == 1:
                circle += 1
            if labels[i][1] == 1:
                triangle += 1
            if labels[i][2] == 1:
                cross += 1
    print('Circle : {}, Triangle : {}, Cross : {}'.format(circle, triangle, cross))
