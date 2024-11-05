from utils import ANCHORS, nms, plot_image, convert_cells_to_bboxes
from models import YOLOv3
import torch
from PIL import Image
import numpy as np

device = torch.device('cpu')

def main():
    model = YOLOv3()
    state = torch.load('/Users/reza/Career/DMLab/AR TRACKING/ar-track/src/ByteTrack/20.pth', weights_only=True, map_location=device)
    model.load_state_dict(state)
    
    image = np.array(Image.open('/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/images/000034.jpg').resize((1024, 1024)), dtype=np.float32) / 255
    x = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        yhat = model(x)
        
    IMAGE_SIZE = 1024
    GRID_SIZES = [IMAGE_SIZE//32, IMAGE_SIZE//16, IMAGE_SIZE//8]
        
    scaled_anchors = (
        torch.tensor(ANCHORS) *
        torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    )
    
    boxes = []
    for i in range(3):
        anchor = scaled_anchors[i]
        # print(anchor)
        yy = yhat[i]
        print(yy.shape)
        boxes += convert_cells_to_bboxes(
                yy, is_predictions=True, s=yhat[i].shape[2], anchors=anchor 
                )[0]
    boxes = nms(boxes, iou_threshold=.3, threshold=.7)
    plot_image(x[0][0], boxes)

if __name__ == '__main__':
    main()
