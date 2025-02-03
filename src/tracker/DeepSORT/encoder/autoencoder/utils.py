import torch
import cv2 as cv
import numpy as np
import pandas as pd
from os import listdir
from astropy.io import fits
from os.path import join as path_join


_mask_rad = 1900
MASK = np.zeros((4096, 4096), dtype=np.uint8)
cv.circle(MASK, (2048, 2048), _mask_rad, 1, thickness=-1)
MASK = MASK.astype(bool)


def closest_divisible_by_4(n):
    if n % 4 == 0:
        return n
    else:
        remainder = n % 4
        lower = n - remainder
        higher = lower + 4
        return lower if (n - lower) <= (higher - n) else higher


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
        self, image_dir, labels_path, old_size=4096, new_size=1024, dataset_size=10_000
    ):
        w, h = [], []
        label_names = sorted(listdir(labels_path))[:dataset_size]
        for label_name in label_names:
            df = pd.read_csv(path_join(labels_path, label_name))
            for _, row in df.iterrows():
                width, height = (
                    row["width"],
                    row["height"],
                )
                w.append(width)
                h.append(height)
        mean_w, mean_h = np.mean(w), np.mean(h)
        patch_dim = max(mean_h, mean_w) / old_size
        self.real_dim = int(patch_dim * new_size)
        if self.real_dim % 4 != 0:
            self.real_dim = closest_divisible_by_4(self.real_dim)
        print("patch size:", self.real_dim)
        half_dim = self.real_dim // 2
        self.images = []
        image_names = sorted(listdir(image_dir))[:dataset_size]
        for label_name, image_name in zip(label_names, image_names):
            df = pd.read_csv(path_join(labels_path, label_name))
            hdul = fits.open(path_join(image_dir, image_name))
            data = hdul[1].data
            image = preprocess_log_minmax(data, new_size)
            for _, row in df.iterrows():
                min_x, min_y, width, height = (
                    int(row["min_x"] / old_size * new_size),
                    int(row["min_y"] / old_size * new_size),
                    int(row["width"] / old_size * new_size),
                    int(row["height"] / old_size * new_size),
                )
                half_width, half_height = width // 2, height // 2
                x, y = min_x + half_width, min_y + half_height
                patch = image[y - half_dim : y + half_dim, x - half_dim : x + half_dim]
                shape_0, shape_1 = patch.shape
                if shape_0 != self.real_dim or shape_1 != self.real_dim:
                    continue
                self.images.append(patch)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        img = torch.tensor(self.images[ix], dtype=torch.float32).unsqueeze(0)
        return img


if __name__ == "__main__":
    dataset = Dataset(
        "/Users/reza/Career/DMLab/AR_TRACKING/ar-track/data/HEK-JSOC/2024_test/images",
        "/Users/reza/Career/DMLab/AR_TRACKING/ar-track/data/HEK-JSOC/2024_test/labels",
    )
