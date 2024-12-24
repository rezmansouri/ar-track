import os
import torch
import cv2 as cv
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches

_mask_rad = 1950
MASK = np.zeros((4096, 4096), dtype=np.uint8)
cv.circle(MASK, (2048, 2048), _mask_rad, 1, thickness=-1)
MASK = MASK.astype(bool)


def preprocess_log_minmax(los_magnetogram, size=4096):
    """
    todo
    """
    nan_ix = np.isnan(los_magnetogram)
    los_magnetogram[nan_ix] = np.average(los_magnetogram[~nan_ix])
    biased_data = np.abs(los_magnetogram) + 1

    log_scaled_data = np.log(biased_data)

    log_scaled_data[los_magnetogram < 0] *= -1
    final = np.zeros_like(los_magnetogram, dtype=np.float32)
    x_min, x_max = log_scaled_data[MASK].min(), log_scaled_data[MASK].max()
    final[MASK] = (log_scaled_data[MASK] - x_min) / (x_max - x_min)
    if size != 4096:
        final = np.array(final * 255, dtype=np.uint8)
        final = cv.resize(final, (size, size), interpolation=cv.INTER_CUBIC)
        final = np.array(final, dtype=np.float32) / 255
    return final


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, images_dir, labels_dir, anchors, image_size=4096, grid_sizes=[13, 26, 52]
    ):
        self.labels = []
        label_names = sorted(os.listdir(labels_dir))
        for label_name in label_names:
            df = pd.read_csv(os.path.join(labels_dir, label_name))
            label = []
            for _, row in df.iterrows():
                min_x, min_y, width, height = (
                    row["min_x"],
                    row["min_y"],
                    row["width"],
                    row["height"],
                )
                center_x, center_y = min_x + width / 2, min_y + height / 2
                label.append(
                    (center_x / 4096, center_y / 4096, width / 4096, height / 4096)
                )
            self.labels.append(label)
        self.images_dir = images_dir
        self.images_names = sorted(os.listdir(images_dir))
        self.image_size = image_size
        self.grid_sizes = grid_sizes
        self.anchors = anchors.reshape(-1, 2)
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images_names[idx])
        print(img_path)
        hdul = fits.open(img_path)
        data = hdul[1].data
        image = preprocess_log_minmax(data, self.image_size)
        image = np.expand_dims(image, 0)
        targets = [
            torch.zeros((self.num_anchors_per_scale, s, s, 5)) for s in self.grid_sizes
        ]
        bboxes = self.labels[idx]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, is_pred=False)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height = box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                s = self.grid_sizes[scale_idx]
                i, j = int(s * y), int(s * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = s * x - j, s * y - i
                    width_cell, height_cell = (width * s, height * s)
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    has_anchor[scale_idx] = True
                elif (
                    not anchor_taken
                    and iou_anchors[anchor_idx] > self.ignore_iou_thresh
                ):
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1
        return image, tuple(targets)
