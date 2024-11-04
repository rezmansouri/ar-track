from utils import ANCHORS, Dataset, nms, plot_image, convert_cells_to_bboxes
import torch
import numpy as np

def main():
    dataset = Dataset('/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/images',
                      '/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/labels.csv',
                      ANCHORS)
    x, y = dataset[0]
    boxes = []
    for i in range(3):
        anchor = ANCHORS[i]
        yy = torch.unsqueeze(y[i], 0)
        boxes += convert_cells_to_bboxes(
                yy, is_predictions=False, s=y[i].shape[2], anchors=anchor
                )[0]
    boxes = nms(boxes, iou_threshold=1, threshold=0.7)
    plot_image(x[0], boxes)

if __name__ == '__main__':
    main()
