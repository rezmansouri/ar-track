import os
from tqdm import tqdm
from sort import Sort
from utils import ANCHORS, nms, save_image, convert_cells_to_bboxes
from models import YOLOv3
import torch
from PIL import Image
import numpy as np
from io import BytesIO
import imageio.v2 as imageio

device = torch.device('cpu')

def main():
    model = YOLOv3()
    state = torch.load('/Users/reza/Career/DMLab/AR TRACKING/ar-track/src/yolo_v3.pth', weights_only=True, map_location=device)
    model.load_state_dict(state)
            
    IMAGE_SIZE = 1024
    GRID_SIZES = [IMAGE_SIZE//32, IMAGE_SIZE//16, IMAGE_SIZE//8]
        
    scaled_anchors = (
        torch.tensor(ANCHORS) *
        torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    )
    
    tracker = Sort()
    
    frames = []
    for file in tqdm(sorted(os.listdir('/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/images'), key=lambda x: int(x[:-4]))[:500]):
        image = np.array(Image.open(os.path.join('/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/images', file)).resize((1024, 1024)), dtype=np.float32) / 255
        x = torch.tensor(image).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            yhat = model(x)
        
        boxes = []
        for i in range(3):
            anchor = scaled_anchors[i]
            yy = yhat[i]
            boxes += convert_cells_to_bboxes(
                    yy, is_predictions=True, s=yhat[i].shape[2], anchors=anchor
                    )[0]
        boxes = nms(boxes, iou_threshold=.3, threshold=.7)
        boxes = np.array(boxes)
        if len(boxes) != 0:
            boxes = np.hstack([boxes[:, 1:], boxes[:, [0]]])
            boxes[:, 0:2] -= boxes[:, 2:4] / 2
            boxes[:, 2:4] += boxes[:, 0:2]
            trks = tracker.update(boxes)
            buf = BytesIO()
            buf = save_image(x[0][0], trks, buf)
            frames.append(imageio.imread(buf))
        
    output_filename = f"out.mp4"
    fps = 30  # Adjust frames per second as needed

    # Create video directly from frames in memory
    with imageio.get_writer(output_filename, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"Video saved as {output_filename}")
    
    # image = np.array(Image.open('/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/images/000200.jpg').resize((1024, 1024)), dtype=np.float32) / 255
    # x_1 = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    
    # with torch.no_grad():
    #     yhat = model(x_1)
    
    # boxes = []
    # for i in range(3):
    #     anchor = scaled_anchors[i]
    #     yy = yhat[i]
    #     boxes += convert_cells_to_bboxes(
    #             yy, is_predictions=True, s=yhat[i].shape[2], anchors=anchor)[0]
    # boxes_1 = nms(boxes, iou_threshold=.3, threshold=.7)
    # plot_image(x_1[0][0], boxes_1, '1')
    
    
    # image = np.array(Image.open('/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/images/000202.jpg').resize((1024, 1024)), dtype=np.float32) / 255
    # x_2 = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    
    # with torch.no_grad():
    #     yhat = model(x_2)
    
    # boxes = []
    # for i in range(3):
    #     anchor = scaled_anchors[i]
    #     yy = yhat[i]
    #     boxes += convert_cells_to_bboxes(
    #             yy, is_predictions=True, s=yhat[i].shape[2], anchors=anchor)[0]
    # boxes_2 = nms(boxes, iou_threshold=.3, threshold=.7)
    # plot_image(x_2[0][0], boxes_2, '2')
    
    
    # # TRACKING
    # tracker = Sort()
    # boxes_1 = np.array(boxes_1)
    # boxes_1[:, [0, 4]] = boxes_1[:, [4, 0]]
    # boxes_1[:, 2:4] += boxes_1[:, 1:3]
    # boxes_2 = np.array(boxes_2)
    # boxes_2[:, 2:4] += boxes_2[:, 1:3]
    # boxes_2[:, [0, 4]] = boxes_2[:, [4, 0]]
    
    # trks = tracker.update(boxes_1)
    # print(trks)
    # trks = tracker.update(boxes_2)
    # print(trks)



if __name__ == '__main__':
    main()
