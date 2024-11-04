import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


ANCHORS = np.array([
       [[ 89.75458716, 106.92431193],
        [174.64      , 158.47703704],
        [289.96757458, 187.5693904 ]],
       [[221.38817481, 271.33161954],
        [358.9468599 , 263.64251208],
        [500.21971253, 321.72689938]],
       [[372.8490566 , 432.9509434 ],
        [639.91686461, 471.93349169],
        [961.578125  , 637.625     ]]])


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
            (pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]),
        )

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(pred[..., 1:3]),
                            torch.exp(pred[..., 3:5]) * anchors
                            ],dim=-1)
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]),
                            ious * target[..., 0:1][obj])
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        box_loss = self.mse(pred[..., 1:5][obj],
                            target[..., 1:5][obj])
        return (
            box_loss
            + object_loss
            + no_object_loss
        )


class Dataset(torch.utils.data.Dataset):
   
    def __init__(
        self, image_dir, labels_path, anchors,
        image_size=416, grid_sizes=[13, 26, 52], original_image_size=4096
    ):
        labels = []
        with open(labels_path, 'r', encoding='utf-8') as label_file:
            label_file.readline()
            label = []
            ix_prev = '1'
            for line in label_file.readlines():
                ix, x1, y1, w, h = [a for a in line.split(',')]
                x1, y1, w, h = [float(a) / original_image_size for a in [x1, y1, w, h]]
                x = x1 + w / 2
                y = y1 + h / 2
                if ix != ix_prev:
                    labels.append(label)
                    label = [[x, y, w, h]]
                    ix_prev = ix
                else:
                    label.append([x, y, w, h])
        self.labels = labels
        self.image_dir = image_dir
        self.image_size = image_size
        self.grid_sizes = grid_sizes
        self.anchors = anchors.reshape(-1, 2) / original_image_size
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5
       
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, str(idx+1).zfill(6) + '.jpg')
        image = np.expand_dims(np.array(Image.open(img_path).resize((416, 416)), dtype=np.float32) / 255., 0)
        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 5))
                for s in self.grid_sizes]
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
                                        [x_cell, y_cell, width_cell,
                                        height_cell]
                                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1
        return image, tuple(targets)

def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True):
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:]
    if is_predictions:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(
            box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
    else:
        scores = predictions[..., 0:1]
    cell_indices = (
        torch.arange(s)
        .repeat(predictions.shape[0], 3, s, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / s * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / s * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    width_height = 1 / s * box_predictions[..., 2:4]
    converted_bboxes = torch.cat(
        (scores, x, y, width_height), dim=-1
    ).reshape(batch_size, num_anchors * s * s, 5)
    return converted_bboxes.tolist()


def plot_image(image, boxes):
    img = np.array(image)
    h, w = img.shape
    _, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')
    for box in boxes:
       
        box = box[1:]
       
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
       
        rect = patches.Rectangle(
            (upper_left_x * w, upper_left_y * h),
            box[2] * w,
            box[3] * h,
            linewidth=2,
            edgecolor='red',
            facecolor="none",
        )
       
        ax.add_patch(rect)
    plt.show()

def nms(bboxes, iou_threshold, threshold):
	
    bboxes = [box for box in bboxes if box[0] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    bboxes_nms = []
    while bboxes:
        first_box = bboxes.pop(0)
        for box in bboxes:
            if box[0] != first_box[0] or iou(
                torch.tensor(first_box[1:]),
                torch.tensor(box[1:]),
            ) < iou_threshold:
               
                if box not in bboxes_nms:
                   
                    bboxes_nms.append(box)
    return bboxes_nms

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
       
        return iou_score.float()
    else:
        box, boxes = box1, box2
        x = np.minimum(box[0], boxes[:, 0])
        y = np.minimum(box[1], boxes[:, 1])
        intersection = x * y
        box_area = box[0] * box[1]
        boxes_area = boxes[:, 0] * boxes[:, 1]
        union = box_area + boxes_area - intersection
        return (intersection / union).float()