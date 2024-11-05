from utils import ANCHORS, nms, save_image, convert_cells_to_bboxes
from models import YOLOv3
import torch
from PIL import Image
import numpy as np
import imageio.v2 as imageio
from io import BytesIO
from tqdm import tqdm
import os

device = torch.device('cpu')

def main():
    model = YOLOv3()
    state = torch.load('/Users/reza/Career/DMLab/AR TRACKING/ar-track/src/ByteTrack/20.pth', weights_only=True, map_location=device)
    model.load_state_dict(state)
        
    IMAGE_SIZE = 1024
    GRID_SIZES = [IMAGE_SIZE//32, IMAGE_SIZE//16, IMAGE_SIZE//8]
        
    scaled_anchors = (
        torch.tensor(ANCHORS) *
        torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    )
    
    thresh, io_thresh = .7, .3
    
    frames = []
    for file in tqdm(sorted(os.listdir('/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/images'), key=lambda x: int(x[:-4]))):
        image = np.array(Image.open(os.path.join('/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/images', file)).resize((1024, 1024)), dtype=np.float32) / 255
        x = torch.tensor(image).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            yhat = model(x)
        
        boxes = []
        for i in range(3):
            anchor = scaled_anchors[i]
            # print(anchor)
            yy = yhat[i]
            # print(yy.shape)
            boxes += convert_cells_to_bboxes(
                    yy, is_predictions=True, s=yhat[i].shape[2], anchors=anchor 
                    )[0]
        boxes = nms(boxes, iou_threshold=io_thresh, threshold=thresh)
        buf = BytesIO()
        buf = save_image(x[0][0], boxes, buf)
        frames.append(imageio.imread(buf))
        
    output_filename = f"conf_{thresh}-iou_{io_thresh}.mp4"
    fps = 30  # Adjust frames per second as needed

    # Create video directly from frames in memory
    with imageio.get_writer(output_filename, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"Video saved as {output_filename}")

if __name__ == '__main__':
    main()
