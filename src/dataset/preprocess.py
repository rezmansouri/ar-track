"""
todo
"""

import cv2 as cv
import numpy as np

_mask_rad = 1950
MASK = np.zeros((4096, 4096), dtype=np.uint8)
cv.circle(MASK, (2048, 2048), _mask_rad, 1, thickness=-1)
MASK = MASK.astype(bool)


def preprocess_log_minmax(los_magnetogram, size=4096):
    """
    todo
    """

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


def preprocess_log_z_minmax(los_magnetogram, size=4096):
    """
    todo
    """
    biased_data = np.abs(los_magnetogram) + 1

    log_scaled_data = np.log(biased_data)

    log_scaled_data[los_magnetogram < 0] *= -1

    mean_value = np.mean(log_scaled_data[MASK])
    std_dev = np.std(log_scaled_data[MASK])
    final = np.zeros_like(los_magnetogram, dtype=np.float32)
    log_scaled_data[MASK] = (log_scaled_data[MASK] - mean_value) / std_dev
    x_min, x_max = log_scaled_data[MASK].min(), log_scaled_data[MASK].max()
    final[MASK] = (log_scaled_data[MASK] - x_min) / (x_max - x_min)
    if size != 4096:
        final = np.array(final * 255, dtype=np.uint8)
        final = cv.resize(final, (size, size), interpolation=cv.INTER_CUBIC)
        final = np.array(final, dtype=np.float32) / 255
    return final


def preprocess_fourfivehunder_log_z_minmax(los_magnetogram, size=4096):
    """
    todo
    """
    los_magnetogram[los_magnetogram > 4500] = 0
    los_magnetogram[los_magnetogram < -4500] = 0
    biased_data = np.abs(los_magnetogram) + 1

    log_scaled_data = np.log(biased_data)

    log_scaled_data[los_magnetogram < 0] *= -1

    mean_value = np.mean(log_scaled_data[MASK])
    std_dev = np.std(log_scaled_data[MASK])
    final = np.zeros_like(los_magnetogram, dtype=np.float32)
    log_scaled_data[MASK] = (log_scaled_data[MASK] - mean_value) / std_dev
    x_min, x_max = log_scaled_data[MASK].min(), log_scaled_data[MASK].max()
    final[MASK] = (log_scaled_data[MASK] - x_min) / (x_max - x_min)
    if size != 4096:
        final = np.array(final * 255, dtype=np.uint8)
        final = cv.resize(final, (size, size), interpolation=cv.INTER_CUBIC)
        final = np.array(final, dtype=np.float32) / 255
    return final
