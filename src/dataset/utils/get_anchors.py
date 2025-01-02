import os
import sys
import numpy as np
import pandas as pd


def one_vs_all_iou(anch_box, boxes):
    x = np.minimum(anch_box[0], boxes[:, 0])
    y = np.minimum(anch_box[1], boxes[:, 1])
    intersection = x * y
    anch_box_area = anch_box[0] * anch_box[1]
    boxes_area = boxes[:, 0] * boxes[:, 1]
    union = anch_box_area + boxes_area - intersection
    return intersection / union


def distance(point, points):
    return 1 - one_vs_all_iou(point, points)


def kmeans(samples, n_clusters, distance_func):
    n_samples = samples.shape[0]
    distances = np.empty((n_samples, n_clusters))
    last_clusters = np.zeros((n_samples))
    nearest_clusters = np.full((n_samples), -1)

    clusters = samples[np.random.choice(n_samples, n_clusters, replace=False)]

    while not (last_clusters == nearest_clusters).all():
        last_clusters = nearest_clusters
        for i in range(n_clusters):
            distances[:, i] = distance_func(clusters[i], samples)
        nearest_clusters = np.argmin(distances, axis=1)
        for i in range(n_clusters):
            clusters[i] = np.mean(samples[nearest_clusters == i], axis=0)

    return clusters, nearest_clusters, distances


def main():
    labels_path = sys.argv[1]
    labels_names = os.listdir(labels_path)
    boxes = []
    for label_name in labels_names:
        df = pd.read_csv(os.path.join(labels_path, label_name))
        for _, row in df.iterrows():
            width, height = (
                row["width"],
                row["height"],
            )
            boxes.append([width, height])
    boxes = np.array(boxes, dtype=np.float32)
    clusters, _, _ = kmeans(boxes, 9, distance_func=distance)
    anchors = sorted(clusters, key=lambda x: x[0] * x[1])
    anchors = np.array(anchors).reshape(3, 3, 2)
    np.save(f"{labels_path}/../anchors.npy", anchors)
    print(anchors)
    print(f"saved in {labels_path}/../anchors.npy")


if __name__ == "__main__":
    main()
