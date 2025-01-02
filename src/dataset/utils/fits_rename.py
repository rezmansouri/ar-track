import os
import sys
from tqdm import tqdm
from sunpy.map import Map


def main():
    fits_paths = sys.argv[1]
    fits_names = os.listdir(fits_paths)
    for fits_name in tqdm(fits_names):
        old_path = os.path.join(fits_paths, fits_name)
        mag = Map(old_path)
        time = mag.date
        new_path = os.path.join(fits_paths, str(time) + ".fits")
        os.rename(old_path, new_path)


if __name__ == "__main__":
    main()
