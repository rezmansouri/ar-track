import torch
from models import YOLOv3
from trainer import training_loop
from utils import ANCHORS, Dataset, YOLOLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Dataset('../../data/AR-MOT/images',
                    '../../data/AR-MOT/labels.csv',
                    ANCHORS)

scaled_anchors = (
    torch.tensor(ANCHORS) *
    torch.tensor([13, 26, 52]).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
).to(device)


loader = torch.utils.data.DataLoader(dataset, 32)

model = YOLOv3().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = YOLOLoss().to(device)

for e in range(1, 20):
    print("Epoch:", e)
    training_loop(loader, model, optimizer, loss_fn, scaled_anchors, device)

    torch.save(model.state_dict(), f'{e}.pth')
