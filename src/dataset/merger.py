import os
import shutil
from datetime import datetime


def main():
    for folder in os.listdir("."):
        if "2024" in folder:
            images_path = os.path.join(folder, "images")
            labels_path = os.path.join(folder, "labels")
            images_names = sorted(os.listdir(images_path))
            labels_names = sorted(os.listdir(labels_path))
            if len(images_names) == len(labels_names) and len(images_names) != 0:
                merged_name = str(datetime.now())
                os.mkdir(merged_name)
                os.mkdir(merged_name, "images")
                os.mkdir(merged_name, "labels")
                for image_name, label_name in zip(images_names, labels_names):
                    shutil.copy(
                        os.path.join(images_path, image_name),
                        os.path.join(merged_name, "images", image_name),
                    )
                    shutil.copy(
                        os.path.join(labels_path, label_name),
                        os.path.join(merged_name, "labels", label_name),
                    )
                print(folder, "done")
            else:
                print(folder, "incomplete")


if __name__ == "__main__":
    main()
