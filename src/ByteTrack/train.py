import torch
from models import YOLOv3
from trainer import training_loop
from utils import ANCHORS, Dataset, YOLOLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 1024
GRID_SIZES = [IMAGE_SIZE//32, IMAGE_SIZE//16, IMAGE_SIZE//8]

dataset = Dataset('../../data/AR-MOT/images',
                    '../../data/AR-MOT/labels.csv',
                    ANCHORS, image_size=IMAGE_SIZE, grid_sizes=GRID_SIZES)

scaled_anchors = (
    torch.tensor(ANCHORS) *
    torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
).to(device)


loader = torch.utils.data.DataLoader(dataset, 1)

model = YOLOv3().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = YOLOLoss().to(device)

for e in range(1, 20):
    print("Epoch:", e)
    training_loop(loader, model, optimizer, loss_fn, scaled_anchors, device)

    torch.save(model.state_dict(), f'{e}.pth')
