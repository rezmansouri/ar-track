from utils import ANCHORS, Dataset, nms, plot_image, convert_cells_to_bboxes
from models import YOLOv3
import torch
from PIL import Image
import numpy as np

device = torch.device('cpu')

def main():
    image = np.array(Image.open('../../data/AR-MOT/images/000010.jpg').resize((1024, 1024)), dtype=np.float32) / 255
    x = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    model = YOLOv3()
    state = torch.load('18.pth', weights_only=True, map_location=device)
    model.load_state_dict(state)
    yhat = model(x)
    
    boxes = []
    for i in range(3):
        anchor = ANCHORS[i]
        yy = torch.unsqueeze(yhat[i], 0)
        boxes += convert_cells_to_bboxes(
                yy, is_predictions=False, s=yhat[i].shape[2], anchors=anchor
                )[0]
        
    boxes = nms(boxes, iou_threshold=1, threshold=0.3)
    print(boxes)
    plot_image(x[0][0], boxes)

if __name__ == '__main__':
    main()
