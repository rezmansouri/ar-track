import os
from tqdm import tqdm
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from utils import ANCHORS, nms, save_image, convert_cells_to_bboxes
from models import YOLOv3
import torch
from PIL import Image
import numpy as np
from io import BytesIO
import imageio.v2 as imageio
import argparse
from encoder import model as encoder_models

device = torch.device("cpu")


def main(max_cosine_distance, nn_budget):
    model = YOLOv3()
    state = torch.load(
        "/Users/reza/Career/DMLab/AR TRACKING/ar-track/src/yolo_v3.pth",
        weights_only=True,
        map_location=device,
    )
    model.load_state_dict(state)

    encoder = encoder_models.Encoder()
    state = torch.load(
        "/Users/reza/Career/DMLab/AR TRACKING/ar-track/src/tracker/DeepSORT/enc.pth",
        weights_only=True,
        map_location=device,
    )
    encoder.load_state_dict(state)

    IMAGE_SIZE = 1024
    GRID_SIZES = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
    PATCH_DIM = 0.08294
    real_dim = int(PATCH_DIM * 1024)
    half_dim = real_dim // 2

    scaled_anchors = torch.tensor(ANCHORS) * torch.tensor(GRID_SIZES).unsqueeze(
        1
    ).unsqueeze(1).repeat(1, 3, 2)

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = Tracker(metric)

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
        boxes = np.array(boxes)
        if len(boxes) != 0:
            boxes = np.hstack([boxes[:, 1:], boxes[:, [0]]])
            boxes[:, 0:2] -= boxes[:, 2:4] / 2
            # boxes[:, 2:4] += boxes[:, 0:2]

            feats = []
            final_boxes = []
            for box in boxes:
                xx, yy = box[0] + box[2] // 2, box[1] + box[3] // 2
                xx, yy = int(xx * IMAGE_SIZE), int(yy * IMAGE_SIZE)
                real_obj = x[
                    :,
                    :,
                    yy - half_dim : yy + half_dim,
                    xx - half_dim : xx + half_dim
                ]
                if real_obj.shape[2] != real_dim or real_obj.shape[3] != real_dim:
                    continue
                feat = encoder(real_obj)[0]
                feats.append(feat)
                final_boxes.append(box)
            dets = [Detection(b[:4], b[4], feats[i].detach().numpy()) for i, b in enumerate(final_boxes)]

            tracker.predict()
            tracker.update(dets)

            buf = BytesIO()
            buf = save_image(x[0][0], tracker.tracks, buf)
            frames.append(imageio.imread(buf))

    output_filename = f"out.mp4"
    fps = 20  # Adjust frames per second as needed

    # Create video directly from frames in memory
    with imageio.get_writer(output_filename, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"Video saved as {output_filename}")


def bool_string(input_string):
    if input_string not in {"True", "False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return input_string == "True"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deep SORT")
    # parser.add_argument(
    #     "--sequence_dir", help="Path to MOTChallenge sequence directory",
    #     default=None, required=True)
    # parser.add_argument(
    #     "--detection_file", help="Path to custom detections.", default=None,
    #     required=True)
    # parser.add_argument(
    #     "--output_file", help="Path to the tracking output file. This file will"
    #     " contain the tracking results on completion.",
    #     default="/tmp/hypotheses.txt")
    # parser.add_argument(
    #     "--min_confidence", help="Detection confidence threshold. Disregard "
    #     "all detections that have a confidence lower than this value.",
    #     default=0.8, type=float)
    # parser.add_argument(
    #     "--min_detection_height", help="Threshold on the detection bounding "
    #     "box height. Detections with height smaller than this value are "
    #     "disregarded", default=0, type=int)
    # parser.add_argument(
    #     "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
    #     "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance",
        help="Gating threshold for cosine distance " "metric (object appearance).",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--nn_budget",
        help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.",
        type=int,
        default=None,
    )
    # parser.add_argument(
    #     "--display", help="Show intermediate tracking results",
    #     default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.max_cosine_distance, args.nn_budget)
