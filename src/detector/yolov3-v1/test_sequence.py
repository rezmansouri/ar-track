from utils import nms, save_image, convert_cells_to_bboxes, preprocess_log_minmax
from models import YOLOv3
import torch
from astropy.io import fits
import numpy as np
import imageio.v2 as imageio
from io import BytesIO
from tqdm import tqdm
import os

device = torch.device("cpu")

MAG_SIZE = 4096
IMAGE_SIZE = 1024
GRID_SIZES = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]


def main():
    state_path = "9.pth"
    anchors_path = "/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/HEK-JSOC/2024_test/anchors.npy"
    images_path = (
        "/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/HEK-JSOC/2024_test/images"
    )
    labels_path = (
        "/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/HEK-JSOC/2024_test/labels"
    )
    model = YOLOv3()
    state = torch.load(state_path, weights_only=True, map_location=device)
    model.load_state_dict(state)

    anchors = np.load(anchors_path) / MAG_SIZE
    scaled_anchors = torch.tensor(anchors) * torch.tensor(GRID_SIZES).unsqueeze(
        1
    ).unsqueeze(1).repeat(1, 3, 2)

    thresh, io_thresh = 0.7, 0.1

    frames = []
    for file in tqdm(sorted(os.listdir(images_path))):
        img_path = os.path.join(images_path, file)
        hdul = fits.open(img_path)
        data = hdul[1].data
        image = preprocess_log_minmax(data, IMAGE_SIZE)
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
    fps = 15  # Adjust frames per second as needed

    # Create video directly from frames in memory
    with imageio.get_writer(output_filename, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"Video saved as {output_filename}")


if __name__ == "__main__":
    main()
