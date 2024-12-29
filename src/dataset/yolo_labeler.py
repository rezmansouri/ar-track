import os
import sys
import pandas as pd
from datetime import datetime


def main():
    hek_labels_path = sys.argv[1]
    hek_labels_names = os.listdir(hek_labels_path)
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    os.mkdir(f"{now}-yolo-labels")
    for hek_label_name in hek_labels_names:
        df = pd.read_csv(os.path.join(hek_labels_path, hek_label_name))
        if len(df) == 0:
            pass
        with open(
            os.path.join(f"{now}-yolo-labels", hek_label_name[:-3] + "txt"),
            "w",
            encoding="utf-8",
        ) as txt_label:
            for _, row in df.iterrows():
                min_x, min_y, width, height = (
                    row["min_x"],
                    row["min_y"],
                    row["width"],
                    row["height"],
                )
                center_x, center_y = min_x + width / 2, min_y + height / 2
                center_x, center_y, min_x, min_y = [
                    i / 4096 for i in [center_x, center_y, width, height]
                ]
                txt_label.write(f"0 {center_x} {center_y} {width} {height}")


if __name__ == "__main__":
    main()
