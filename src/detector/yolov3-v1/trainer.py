from tqdm import tqdm

def training_loop(train_loader, model, optimizer, loss_fn, scaled_anchors, device):
    progress_bar = tqdm(train_loader, leave=True)

    losses = []

    for _, (x, y) in enumerate(progress_bar):
        optimizer.zero_grad()
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

        losses.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        mean_loss = sum(losses) / len(losses)
        progress_bar.set_postfix(loss=mean_loss)
    return mean_loss