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
        "/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/images",
        "/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/labels.csv",
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    N_EPOCHS = 1

    encoder = model.Encoder().to(device)
    decoder = model.Decoder().to(device)
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
        os.path.join('.', "losses", "train.npy"),
        np.array(train_losses, dtype=np.float32),
    )
    
    torch.save(enc.state_dict(), f'enc.pth')
    torch.save(dec.state_dict(), f'dec.pth')


if __name__ == "__main__":
    main()
