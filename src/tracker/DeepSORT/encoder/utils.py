import os
import torch
import numpy as np
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, image_dir, labels_path, old_size=4096, new_size=1024, patch_dim=0.08294
    ):
        real_dim = int(patch_dim * 1024)
        self.half_dim = real_dim // 2
        self.images = []
        i = 1
        with open(labels_path, "r", encoding="utf-8") as label_file:
            label_file.readline()
            line = label_file.readline()
            while line:
                image_name = str(i).zfill(6) + ".jpg"
                image_path = os.path.join(image_dir, image_name)
                image = (
                    np.array(
                        Image.open(image_path).resize((new_size, new_size)),
                        dtype=np.float32,
                    )
                    / 255.0
                )
                while line:
                    ix, x1, y1, w, h = [a for a in line.split(",")]
                    ix = int(ix)
                    x1, y1, w, h = [float(a) / old_size for a in [x1, y1, w, h]]
                    xx1, yy1 = int(x1 * new_size), int(y1 * new_size)
                    xx, yy = int(xx1 + w * new_size / 2), int(yy1 + h * new_size / 2)
                    self.images.append(
                        image[
                            yy - self.half_dim : yy + self.half_dim,
                            xx - self.half_dim : xx + self.half_dim,
                        ]
                    )
                    if ix != i:
                        i += 1
                        break
                    line = label_file.readline()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        img = torch.tensor(self.images[ix], dtype=torch.float32).unsqueeze(0)
        return img


if __name__ == "__main__":
    dataset = Dataset(
        "/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/images",
        "/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/labels.csv",
    )
