import torch
import numpy as np


def predict(batch, enc, dec, loss_fn, device):
    with torch.no_grad():
        batch = batch.to(device)
        out = enc(batch)
        out = dec(out)
        loss = loss_fn(out, batch)
        return out, loss


def fit(train_loader, enc, dec, loss_fn, optimizer, n_epochs, device):
    train_losses = []
    for epoch in range(1, n_epochs + 1):
        enc.train()
        dec.train()
        train_loss = []
        for batch in train_loader:
            batch = batch.to(device)
            out = enc(batch)
            out = dec(out)
            loss = loss_fn(out, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        train_losses.append(train_loss)

        print(
            f"epoch {epoch:>3}\ttraining loss: {train_loss:0.5f}")

    return train_losses, (enc, dec)
