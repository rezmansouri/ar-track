import os
import sys
import json
import utils
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from model import MobileNetV1
from datetime import datetime
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DETECTOR_IMAGE_DIM = 2048
FINAL_PATCH_DIM = 256
LATENT_DIM = 256
FRAME_JUMP = 24


def main():
    (
        train_images_path,
        train_labels_path,
        val_images_path,
        val_labels_path,
        results_path,
        batch_size,
        n_epochs,
    ) = sys.argv[1:]
    batch_size, n_epochs = int(batch_size), int(n_epochs)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir(f"{results_path}/{time}")
    train_dataset = utils.TrainDataset(
        train_images_path,
        train_labels_path,
        DETECTOR_IMAGE_DIM,
        FINAL_PATCH_DIM,
        frame_jump=FRAME_JUMP,
    )
    train_mean_patch_dim = train_dataset.get_mean_patch_dim()
    results_path = f"{results_path}/{time}"
    with open(f"{results_path}/infer_config.json", "w") as f:
        json.dump(
            {
                "train_images_path": train_images_path,
                "train_labels_path": train_labels_path,
                "val_images_path": val_images_path,
                "val_labels_path": val_labels_path,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "train_mean_patch_dim": train_mean_patch_dim,
                "DETECTOR_IMAGE_DIM": DETECTOR_IMAGE_DIM,
                "FINAL_PATCH_DIM": FINAL_PATCH_DIM,
                "LATENT_DIM": LATENT_DIM,
                "FRAME_JUMP": FRAME_JUMP,
            },
            f,
            indent=4,
        )
    val_dataset = utils.ValDataset(
        val_images_path,
        val_labels_path,
        DETECTOR_IMAGE_DIM,
        train_mean_patch_dim,
        FINAL_PATCH_DIM,
        frame_jump=FRAME_JUMP,
    )
    best_val_loss = torch.inf
    best_state = None
    best_epoch = -1
    train_loss, val_loss = [], []
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = MobileNetV1().to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = utils.triplet_loss

    for epoch in range(1, n_epochs + 1):
        t_loss = []
        model.train()
        for a, p, n in tqdm(train_loader, leave=True):
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            anchor, positive, negative = model(a), model(p), model(n)
            loss = loss_fn(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            t_loss.append(loss.item())
        t_loss = np.mean(t_loss)
        train_loss.append(t_loss)
        model.eval()
        v_loss = []
        for a, p, n in tqdm(val_loader, leave=True):
            a, p, n = a.to(device), p.to(device), n.to(device)
            anchor, positive, negative = model(a), model(p), model(n)
            loss = loss_fn(anchor, positive, negative)
            v_loss.append(loss.item())
        v_loss = np.mean(v_loss)
        val_loss.append(v_loss)
        if v_loss < best_val_loss:
            best_epoch = epoch
            best_state = model.state_dict()
            best_val_loss = v_loss
        val_loss.append(v_loss)
        print(
            f"epoch:{epoch}/{n_epochs}\tvalidation loss{v_loss}\ttraining loss:{t_loss}"
        )

    np.save(f"{results_path}/train_loss.npy", np.array(train_loss))
    np.save(f"{results_path}/val_loss.npy", np.array(val_loss))
    torch.save(best_state, f"{results_path}/{best_epoch}.pth")


if __name__ == "__main__":
    main()
