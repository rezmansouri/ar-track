import os
import sys
import pandas as pd
from ultralytics import YOLO


def main():
    state_path, image_size, batch_size, scale, iou_thresh, conf_thresh, devices = (
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]),
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
        sys.argv[7],
    )
    devices = [int(d) for d in devices]
    model = YOLO(state_path)

    os.makedirs(f"./runs/detect/test/{scale}", exist_ok=True)

    iou = float(iou_thresh)
    conf = float(conf_thresh)
    metrics = model.val(
        data="/home/rmansouri1/ar_track/ar-track/src/detector/yolov11/ultralytics/hek.yaml",
        imgsz=image_size,
        batch=batch_size,
        save_json=True,
        conf=conf,
        iou=iou,
        plots=True,
        split="test",
        device=devices,
        project=f"./runs/detect/test/{scale}",
        name=f"conf-{conf}-iou{iou}",
    )
    print(f"conf: {conf} iou: {iou} mAP50-95: {metrics.box.map}")
    result = {
        "map50-95": [metrics.box.map],
        "map50": [metrics.box.map50],
        "map75": [metrics.box.map75],
    }
    df = pd.DataFrame(result)
    df.to_csv(
        f"./runs/detect/validation/{scale}/conf-{conf}-iou{iou}/maps.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
