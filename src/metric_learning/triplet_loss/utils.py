import os
import torch
import cv2 as cv
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(
        self,
        images_path,
        labels_path,
        detector_image_dim,
        final_patch_dim,
        frame_jump=24,
        org_image_dim=4096,
    ):
        self.frame_jump_thresh = frame_jump if frame_jump > 1 else 2
        self.detector_image_dim = detector_image_dim
        self.final_patch_dim = final_patch_dim
        self.labels_path = labels_path
        self.label_names = sorted(os.listdir(labels_path))
        self.label_paths = [
            os.path.join(labels_path, label_name) for label_name in self.label_names
        ]
        self.images_path = images_path
        self.labels = []
        self.ar_dict = {}
        w, h = [], []
        for label_path in self.label_paths:
            df = pd.read_csv(label_path)
            for _, row in df.iterrows():
                ar_noaanum, min_x, min_y, width, height = (
                    row["ar_noaanum"],
                    row["min_x"] / org_image_dim * self.detector_image_dim,
                    row["min_y"] / org_image_dim * self.detector_image_dim,
                    row["width"] / org_image_dim * self.detector_image_dim,
                    row["height"] / org_image_dim * self.detector_image_dim,
                )
                x, y = min_x + width / 2, min_y + height / 2
                w.append(width)
                h.append(height)
                image_path = os.path.join(
                    self.images_path, os.path.basename(label_path)[:-3] + "jpg"
                )
                if ar_noaanum in self.ar_dict:
                    self.ar_dict[ar_noaanum].append((image_path, int(x), int(y)))
                else:
                    self.ar_dict[ar_noaanum] = [(image_path, int(x), int(y))]
        mean_w, mean_h = np.mean(w), np.mean(h)
        self.half_patch_dim = int(max(mean_h, mean_w) / 2)
        self.arnoaanum_list = list(self.ar_dict.keys())

    def get_mean_patch_dim(self):
        return self.half_patch_dim * 2

    def _extract_patch(self, image, x, y):
        if image.shape != (self.detector_image_dim, self.detector_image_dim):
            image = cv.resize(
                image,
                (self.detector_image_dim, self.detector_image_dim),
                interpolation=cv.INTER_LINEAR,
            )
        x_min, x_max = max(0, x - self.half_patch_dim), min(
            self.detector_image_dim, x + self.half_patch_dim
        )
        y_min, y_max = max(0, y - self.half_patch_dim), min(
            self.detector_image_dim, y + self.half_patch_dim
        )
        patch = image[
            y_min:y_max,
            x_min:x_max,
        ]
        pad_height = max(0, self.half_patch_dim * 2 - patch.shape[0])
        pad_width = max(0, self.half_patch_dim * 2 - patch.shape[1])

        if pad_width > 0:
            if x > self.detector_image_dim // 2:
                patch = np.pad(
                    patch,
                    (
                        (0, 0),
                        (0, pad_width),
                    ),
                    mode="constant",
                    constant_values=0,
                )
            else:
                patch = np.pad(
                    patch,
                    (
                        (0, 0),
                        (pad_width, 0),
                    ),
                    mode="constant",
                    constant_values=0,
                )
        if pad_height > 0:
            if y > self.detector_image_dim // 2:
                patch = np.pad(
                    patch,
                    (
                        (0, pad_height),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0,
                )
            else:
                patch = np.pad(
                    patch,
                    (
                        (pad_height, 0),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0,
                )
        if patch.shape != (self.final_patch_dim, self.final_patch_dim):
            patch = cv.resize(patch, (self.final_patch_dim, self.final_patch_dim))
        patch = np.asarray(patch, dtype=np.float32)
        patch = np.expand_dims(patch, 0)
        return patch

    def __len__(self):
        return len(self.ar_dict)

    def __getitem__(self, idx):
        anchor_noaa = self.arnoaanum_list[idx]
        anchor_idx = np.random.randint(len(self.ar_dict[anchor_noaa]))
        frame_jump = np.random.randint(1, self.frame_jump_thresh)
        positive_idx = min(anchor_idx + frame_jump, len(self.ar_dict[anchor_noaa]) - 1)
        if positive_idx - anchor_idx > self.frame_jump_thresh:
            positive_idx = len(self.ar_dict[anchor_noaa]) // 2
        a_image_path, a_x, a_y = self.ar_dict[anchor_noaa][anchor_idx]
        p_image_path, p_x, p_y = self.ar_dict[anchor_noaa][positive_idx]
        negative_noaa = anchor_noaa
        while negative_noaa == anchor_noaa:
            negative_noaa = np.random.choice(self.arnoaanum_list)
        negative_idx = np.random.randint(len(self.ar_dict[negative_noaa]))
        n_image_path, n_x, n_y = self.ar_dict[negative_noaa][negative_idx]
        a_image, p_image, n_image = (
            cv.imread(a_image_path, cv.IMREAD_GRAYSCALE) / 255.0,
            cv.imread(p_image_path, cv.IMREAD_GRAYSCALE) / 255.0,
            cv.imread(n_image_path, cv.IMREAD_GRAYSCALE) / 255.0,
        )
        a_patch = self._extract_patch(a_image, a_x, a_y)
        p_patch = self._extract_patch(p_image, p_x, p_y)
        n_patch = self._extract_patch(n_image, n_x, n_y)
        return a_patch, p_patch, n_patch


class ValDataset(Dataset):
    def __init__(
        self,
        images_path,
        labels_path,
        detector_image_dim,
        mean_patch_dim,
        final_patch_dim,
        frame_jump=24,
        org_image_dim=4096,
    ):
        self.frame_jump_thresh = frame_jump if frame_jump > 1 else 2
        self.detector_image_dim = detector_image_dim
        self.final_patch_dim = final_patch_dim
        self.labels_path = labels_path
        self.label_names = sorted(os.listdir(labels_path))
        self.label_paths = [
            os.path.join(labels_path, label_name) for label_name in self.label_names
        ]
        self.images_path = images_path
        self.labels = []
        self.ar_dict = {}
        self.half_patch_dim = mean_patch_dim // 2
        for label_path in self.label_paths:
            df = pd.read_csv(label_path)
            for _, row in df.iterrows():
                ar_noaanum, min_x, min_y, width, height = (
                    row["ar_noaanum"],
                    row["min_x"] / org_image_dim * self.detector_image_dim,
                    row["min_y"] / org_image_dim * self.detector_image_dim,
                    row["width"] / org_image_dim * self.detector_image_dim,
                    row["height"] / org_image_dim * self.detector_image_dim,
                )
                x, y = min_x + width / 2, min_y + height / 2
                image_path = os.path.join(
                    self.images_path, os.path.basename(label_path)[:-3] + "jpg"
                )
                if ar_noaanum in self.ar_dict:
                    self.ar_dict[ar_noaanum].append((image_path, int(x), int(y)))
                else:
                    self.ar_dict[ar_noaanum] = [(image_path, int(x), int(y))]
        self.arnoaanum_list = list(self.ar_dict.keys())

    def _extract_patch(self, image, x, y):
        if image.shape != (self.detector_image_dim, self.detector_image_dim):
            image = cv.resize(
                image,
                (self.detector_image_dim, self.detector_image_dim),
                interpolation=cv.INTER_LINEAR,
            )
        x_min, x_max = max(0, x - self.half_patch_dim), min(
            self.detector_image_dim, x + self.half_patch_dim
        )
        y_min, y_max = max(0, y - self.half_patch_dim), min(
            self.detector_image_dim, y + self.half_patch_dim
        )
        patch = image[
            y_min:y_max,
            x_min:x_max,
        ]
        pad_height = max(0, self.half_patch_dim * 2 - patch.shape[0])
        pad_width = max(0, self.half_patch_dim * 2 - patch.shape[1])

        if pad_width > 0:
            if x > self.detector_image_dim // 2:
                patch = np.pad(
                    patch,
                    (
                        (0, 0),
                        (0, pad_width),
                    ),
                    mode="constant",
                    constant_values=0,
                )
            else:
                patch = np.pad(
                    patch,
                    (
                        (0, 0),
                        (pad_width, 0),
                    ),
                    mode="constant",
                    constant_values=0,
                )
        if pad_height > 0:
            if y > self.detector_image_dim // 2:
                patch = np.pad(
                    patch,
                    (
                        (0, pad_height),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0,
                )
            else:
                patch = np.pad(
                    patch,
                    (
                        (pad_height, 0),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0,
                )
        if patch.shape != (self.final_patch_dim, self.final_patch_dim):
            patch = cv.resize(patch, (self.final_patch_dim, self.final_patch_dim))
        patch = np.asarray(patch, dtype=np.float32)
        patch = np.expand_dims(patch, 0)
        return patch

    def __len__(self):
        return len(self.ar_dict)

    def __getitem__(self, idx):
        anchor_noaa = self.arnoaanum_list[idx]
        anchor_idx = np.random.randint(len(self.ar_dict[anchor_noaa]))
        frame_jump = np.random.randint(1, self.frame_jump_thresh)
        positive_idx = min(anchor_idx + frame_jump, len(self.ar_dict[anchor_noaa]) - 1)
        if positive_idx - anchor_idx > self.frame_jump_thresh:
            positive_idx = len(self.ar_dict[anchor_noaa]) // 2
        a_image_path, a_x, a_y = self.ar_dict[anchor_noaa][anchor_idx]
        p_image_path, p_x, p_y = self.ar_dict[anchor_noaa][positive_idx]
        negative_noaa = anchor_noaa
        while negative_noaa == anchor_noaa:
            negative_noaa = np.random.choice(self.arnoaanum_list)
        negative_idx = np.random.randint(len(self.ar_dict[negative_noaa]))
        n_image_path, n_x, n_y = self.ar_dict[negative_noaa][negative_idx]
        a_image, p_image, n_image = (
            cv.imread(a_image_path, cv.IMREAD_GRAYSCALE) / 255.0,
            cv.imread(p_image_path, cv.IMREAD_GRAYSCALE) / 255.0,
            cv.imread(n_image_path, cv.IMREAD_GRAYSCALE) / 255.0,
        )
        a_patch = self._extract_patch(a_image, a_x, a_y)
        p_patch = self._extract_patch(p_image, p_x, p_y)
        n_patch = self._extract_patch(n_image, n_x, n_y)
        return a_patch, p_patch, n_patch


def triplet_loss(a, p, n, margin=0.2):
    a = F.normalize(a, p=2, dim=1)
    p = F.normalize(p, p=2, dim=1)
    n = F.normalize(n, p=2, dim=1)
    pos_sim = torch.sum(a * p, dim=1)
    neg_sim = torch.sum(a * n, dim=1)
    loss = F.relu(neg_sim - pos_sim + margin)
    return loss.mean()
