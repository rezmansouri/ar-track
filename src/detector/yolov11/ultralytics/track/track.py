import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from utils import save_image
from ultralytics import YOLO
import imageio.v2 as imageio
from byte import BYTETracker


def main():
    image_size, devices, state_path, data_path = (
        int(sys.argv[1]),
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
    )
    devices = [int(d) for d in devices]

    model = YOLO(state_path)

    image_names = sorted(os.listdir(data_path))
    image_paths = [os.path.join(data_path, image_name) for image_name in image_names]

    parser = argparse.ArgumentParser()
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

    results = model.predict(image_paths, imgsz=image_size, device=devices, batch=-1)

    frames = []
    for image_path, result in tqdm(zip(image_paths, results)):
        image = np.array(Image.open(image_path))
        infer = result.boxes
        probs = infer.conf
        boxes = infer.xyxyn
        boxes_probs = np.hstack([boxes, probs])
        trks = tracker.update(boxes_probs)
        tracks = []
        for t in trks:
            [x1, y1, w, h] = t.tlwh
            tracks.append([x1, y1, w, h, t.track_id])
        buf = BytesIO()
        buf = save_image(image, np.array(tracks), buf)
        frames.append(imageio.imread(buf))
        
    output_filename = f"byte.mp4"
    fps = 7  # Adjust frames per second as needed

    # Create video directly from frames in memory
    with imageio.get_writer(output_filename, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"Video saved as {output_filename}")


if __name__ == "__main__":
    main()
