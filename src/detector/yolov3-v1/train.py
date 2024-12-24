import os
import sys
import torch
import numpy as np
from models import YOLOv3
from trainer import training_loop
from utils import Dataset, YOLOLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAG_SIZE = 4096
IMAGE_SIZE = 2048
GRID_SIZES = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]


def main():
    data_path = sys.argv[1]
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    train_images_path = os.path.join(train_path, "images")
    train_labels_path = os.path.join(train_path, "labels")

    test_images_path = os.path.join(test_path, "images")
    test_labels_path = os.path.join(test_path, "labels")

    anchors = np.load(os.path.join(train_path, "anchors.npy")) / MAG_SIZE
    train_dataset = Dataset(
        train_images_path,
        train_labels_path,
        anchors,
        image_size=IMAGE_SIZE,
        grid_sizes=GRID_SIZES,
    )

    test_dataset = Dataset(
        test_images_path,
        test_labels_path,
        anchors,
        image_size=IMAGE_SIZE,
        grid_sizes=GRID_SIZES,
    )

    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    batch_size = 4

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)

    model = YOLOv3().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = YOLOLoss().to(device)

    train_losses = []
    test_losses = []
    for e in range(1, 21):
        print("Epoch:", e)
        model.train()
        train_loss = training_loop(
            train_loader, model, optimizer, loss_fn, scaled_anchors, device
        )
        test_loss = []
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y0, y1, y2 = (
                    y[0].to(device),
                    y[1].to(device),
                    y[2].to(device),
                )
                outputs = model(x)
                loss = (
                    loss_fn(outputs[0], y0, scaled_anchors[0])
                    + loss_fn(outputs[1], y1, scaled_anchors[1])
                    + loss_fn(outputs[2], y2, scaled_anchors[2])
                )
                test_loss.append(loss.item())
        train_losses.append(train_loss)
        test_losses.append(sum(test_loss) / len(test_loss))

        torch.save(model.state_dict(), f"{e}.pth")


if __name__ == "__main__":
    main()
