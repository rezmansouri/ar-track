import os
import sys
import torch

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import numpy as np
from models import YOLOv3
from trainer import training_loop
from utils import Dataset, YOLOLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 4096
GRID_SIZES = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]


def main():
    train_path = sys.argv[1]
    images_path = os.path.join(train_path, "images")
    labels_path = os.path.join(train_path, "labels")
    anchors = np.load(os.path.join(train_path, "anchors.npy"))
    # transformations = A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.5),
    #     ToTensorV2()
    # ], bbox_params=A.BboxParams(
    #                     format="yolo",
    #                     min_visibility=0.4,
    #                     label_fields=[]
    #                 ))

    dataset = Dataset(
        images_path,
        labels_path,
        anchors,
        image_size=IMAGE_SIZE,
        grid_sizes=GRID_SIZES,
    )  # , transform=transformations)

    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    loader = torch.utils.data.DataLoader(dataset, 8)

    model = YOLOv3().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = YOLOLoss().to(device)

    for e in range(1, 21):
        print("Epoch:", e)
        training_loop(loader, model, optimizer, loss_fn, scaled_anchors, device)

        torch.save(model.state_dict(), f"{e}.pth")


if __name__ == "__main__":
    main()
