import os
import sys
import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
from astropy.io import fits
from datetime import datetime

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
    final = np.array(final * 255, dtype=np.uint8)
    return final


def convert(fits_path, jpg_path):
    hdul = fits.open(fits_path)
    data = hdul[1].data
    arr = preprocess_log_minmax(data)
    image = Image.fromarray(arr)
    image.save(jpg_path)


def main():
    fits_paths = sys.argv[1]
    fits_names = os.listdir(fits_paths)
    time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    jpgs_paths = os.path.join(fits_paths, "..", f"jpg-{time}")
    os.mkdir(jpgs_paths)
    for fits_name in tqdm(fits_names):
        fits_path = os.path.join(fits_paths, fits_name)
        jpg_path = os.path.join(jpgs_paths, fits_name[:-4] + "jpg")
        convert(fits_path, jpg_path)


if __name__ == "__main__":
    main()
