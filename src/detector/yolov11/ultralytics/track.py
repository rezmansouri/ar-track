import os
import sys
from ultralytics import YOLO


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

    for image_name in image_names:
        image_path = os.path.join(data_path, image_name)
        result = model.predict(image_path, imgsz=image_size, device=devices)
        boxes = result.boxes
        probs = result.probs
        print(boxes)
        print(probs)
        break


if __name__ == "__main__":
    main()
