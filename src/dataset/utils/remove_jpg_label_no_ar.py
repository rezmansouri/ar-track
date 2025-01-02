import os
import sys
import pandas as pd


def main():
    jpgs_path, labels_path = sys.argv[1], sys.argv[2]
    jpg_names = sorted(os.listdir(jpgs_path))
    label_names = sorted(os.listdir(labels_path))
    for jpg_name, label_name in zip(jpg_names, label_names):
        jpg_path = os.path.join(jpgs_path, jpg_name)
        label_path = os.path.join(labels_path, label_name)
        if pd.read_csv(label_path).empty:
            print("removed", jpg_name, label_name)
            os.remove(label_path)
            os.remove(jpg_path)


if __name__ == "__main__":
    main()
