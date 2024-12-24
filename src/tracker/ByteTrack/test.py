import os
from tqdm import tqdm
from byte import BYTETracker
from utils import nms, save_image, convert_cells_to_bboxes, preprocess_log_minmax
from models import YOLOv3
import torch
import argparse
from PIL import Image
import numpy as np
from io import BytesIO
import imageio.v2 as imageio

device = torch.device("cpu")


def main():
    model = YOLOv3()
    state = torch.load(
        "/Users/reza/Career/DMLab/AR TRACKING/ar-track/src/yolo_v3.pth",
        weights_only=True,
        map_location=device,
    )
    model.load_state_dict(state)

    IMAGE_SIZE = 1024
    GRID_SIZES = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

    scaled_anchors = torch.tensor(ANCHORS) * torch.tensor(GRID_SIZES).unsqueeze(
        1
    ).unsqueeze(1).repeat(1, 3, 2)

    parser = argparse.ArgumentParser()
    # parser.add_argument('mot20', type='boolean')
    parser.add_argument(
        "--track_thresh", type=float, default=0.6, help="tracking confidence threshold"
    )
    parser.add_argument(
        "--track_buffer", type=int, default=30, help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.9,
        help="matching threshold for tracking",
    )

    args = parser.parse_args([])

    tracker = BYTETracker(args)

    frames = []
    for file in tqdm(
        sorted(
            os.listdir(
                "/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/images"
            ),
            key=lambda x: int(x[:-4]),
        )[:250]
    ):
        image = (
            np.array(
                Image.open(
                    os.path.join(
                        "/Users/reza/Career/DMLab/AR TRACKING/ar-track/data/AR-MOT/images",
                        file,
                    )
                ).resize((1024, 1024)),
                dtype=np.float32,
            )
            / 255
        )
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
        boxes = nms(boxes, iou_threshold=0.3, threshold=0.7)
        boxes = np.array(boxes, dtype=np.float64)
        if len(boxes) != 0:
            boxes = np.hstack([boxes[:, 1:], boxes[:, [0]]])
            boxes[:, 0:2] -= boxes[:, 2:4] / 2
            boxes[:, 2:4] += boxes[:, 0:2]
            trks = tracker.update(boxes)
            tracks = []
            for t in trks:
                [x1, y1, w, h] = t.tlwh
                tracks.append([x1, y1, w, h, t.track_id])
            buf = BytesIO()
            buf = save_image(x[0][0], np.array(tracks), buf)
            frames.append(imageio.imread(buf))

    output_filename = f"byte.mp4"
    fps = 30  # Adjust frames per second as needed

    # Create video directly from frames in memory
    with imageio.get_writer(output_filename, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"Video saved as {output_filename}")


if __name__ == "__main__":
    main()
