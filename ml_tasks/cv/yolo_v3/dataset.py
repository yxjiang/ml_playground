import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    intersection_over_union as iou,
    # non_max_suppression as nms,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self, csv_file, image_dir, label_dir, anchors,
        kernel_sizes=[13, 26, 52], num_classes=20, transform=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.kernel_sizes = kernel_sizes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all scales
        self.num_anchors_per_scale = self.num_anchors // len(anchors)
        self.num_classes = num_classes
        self.ignore_iou_threshold = 0.5  # ignore prediction if iou < 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # [class, x, y, w, h] -> [x, y, w, h, classes]
        bboxes = np.roll(np.loadtxt(fname=label_path, delimitor=" ", ndmin=2), 4, axis=1).tolist()  
        image_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        # 6 values [p_o, x, y, w, h, c]
        num_values = 6
        # Same number of anchros for each prediction
        targets = [torch.zeros((self.num_anchors // 3, s, s, num_values)) for s in self.kernel_sizes]

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale  # 0, 1, 2
                # identify which anchor on the scale assigned to
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale  
                s = self.kernel_sizes[scale_idx]
                i, j = int(s * y), int(s * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = s * x - j, s * y - i  # both are between [0, 1]
                    width_cell, height_cell = (width * s, height * s)
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                    # TODO: add set anchors of scale idx to True
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_threshold:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore the prediction

        return image, tuple(targets)

