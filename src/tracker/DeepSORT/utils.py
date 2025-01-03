import os
import torch
import cv2 as cv
import numpy as np
import pandas as pd
import torch.nn as nn
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches

cmap = plt.get_cmap("hsv")
COLORS = [cmap(i) for i in np.linspace(0, 1, 200)]
np.random.shuffle(COLORS)
COLORS += COLORS + COLORS

_mask_rad = 1900
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


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target, anchors):
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        no_object_loss = self.bce(
            (pred[..., 0:1][no_obj]),
            (target[..., 0:1][no_obj]),
        )

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat(
            [self.sigmoid(pred[..., 1:3]), torch.exp(pred[..., 3:5]) * anchors], dim=-1
        )
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(
            self.sigmoid(pred[..., 0:1][obj]), ious * target[..., 0:1][obj]
        )
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        box_loss = self.mse(pred[..., 1:5][obj], target[..., 1:5][obj])
        return box_loss + object_loss + no_object_loss


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


def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True):
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:]
    if is_predictions:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
    else:
        scores = predictions[..., 0:1]
    cell_indices = (
        torch.arange(s)
        .repeat(predictions.shape[0], 3, s, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = (box_predictions[..., 0:1] + cell_indices) / s
    y = (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4)) / s
    width_height = box_predictions[..., 2:4] / s
    converted_bboxes = torch.cat((scores, x, y, width_height), dim=-1).reshape(
        batch_size, num_anchors * s * s, 5
    )
    return converted_bboxes.tolist()


def save_image(image, tracks, buf):
    img = np.array(image)
    h, w = img.shape
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.axis("off")
    fig.patch.set_visible(False)
    ax.imshow(img, cmap="gray")
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        box = track.to_tlwh()
        # box[2:4] -= box[0:2]

        upper_left_x = box[0]
        upper_left_y = box[1]

        rect = patches.Rectangle(
            (upper_left_x * w, upper_left_y * h),
            box[2] * w,
            box[3] * h,
            linewidth=2,
            edgecolor=COLORS[track.track_id],
            facecolor="none",
        )

        ax.add_patch(rect)
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf


def plot_image(image, boxes, name):
    img = np.array(image)
    h, w = img.shape
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.axis("off")
    fig.patch.set_visible(False)
    ax.imshow(img, cmap="gray")
    for box in boxes:

        box = box[1:]

        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (upper_left_x * w, upper_left_y * h),
            box[2] * w,
            box[3] * h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )

        ax.add_patch(rect)
    plt.savefig(f"{name}.png", format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def nms(bboxes, iou_threshold, threshold):
    # print(len(bboxes))
    bboxes = [box for box in bboxes if box[0] > threshold]
    # print(len(bboxes))
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    nms_bboxes = []
    while bboxes:
        # print(len(bboxes))
        chosen_box = bboxes.pop(0)
        nms_bboxes.append(chosen_box)
        bboxes = [
            box
            for box in bboxes
            if iou(torch.tensor(chosen_box[1:]), torch.tensor(box[1:])) < iou_threshold
        ]
    return nms_bboxes


def iou(box1, box2, is_pred=True):
    if is_pred:
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        union = box1_area + box2_area - intersection

        epsilon = 1e-6
        iou_score = intersection / (union + epsilon)

        return iou_score
    else:
        box, boxes = box1, box2
        x = np.minimum(box[0], boxes[:, 0])
        y = np.minimum(box[1], boxes[:, 1])
        intersection = x * y
        box_area = box[0] * box[1]
        boxes_area = boxes[:, 0] * boxes[:, 1]
        union = box_area + boxes_area - intersection
        return intersection / union
