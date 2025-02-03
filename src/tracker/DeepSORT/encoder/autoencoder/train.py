import os
import utils
import torch
import model
import trainer
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """empty"""
    train_dataset = utils.Dataset(
        "/Users/reza/Career/DMLab/AR_TRACKING/ar-track/data/HEK-JSOC/2024_test/images",
        "/Users/reza/Career/DMLab/AR_TRACKING/ar-track/data/HEK-JSOC/2024_test/labels",
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    dim = train_dataset.real_dim

    N_EPOCHS = 1

    encoder = model.Encoder(input_dim=dim).to(device)
    decoder = model.Decoder(input_dim=dim).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params)
    criterion = torch.nn.BCELoss().to(device)

    optimizer = optim.Adam(params, lr=1e-3)
    optimizer.param_groups[0]["params"] = [
        param.to(device) for param in optimizer.param_groups[0]["params"]
    ]

    train_losses, (enc, dec) = trainer.fit(
        train_loader,
        encoder,
        decoder,
        criterion,
        optimizer,
        N_EPOCHS,
        device,
    )

    np.save(
        "losses.npy",
        np.array(train_losses, dtype=np.float32),
    )


if __name__ == "__main__":
    main()
