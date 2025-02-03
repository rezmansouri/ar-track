import os
import sys
import json
import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from encoder.siamese import model as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    (
        detection_results_path,
        encoder_state_path,
        encoder_training_cfg_path,
        dataset_path,
    ) = sys.argv[1:]
    with open(encoder_training_cfg_path, "r", encoding="utf-8") as f:
        encoder_training_config = json.load(f)
    result_name = os.path.basename(detection_results_path)[:-5]
    print(result_name)

    model = models.MobileNetV1(encoder_training_config["LATENT_DIM"]).to(device)

    model.load_state_dict(torch.load(encoder_state_path, map_location="cpu"))

    model.eval()

    train_mean_patch_dim = encoder_training_config["train_mean_patch_dim"]

    final_patch_dim = encoder_training_config["FINAL_PATCH_DIM"]

    detector_image_dim = encoder_training_config["DETECTOR_IMAGE_DIM"]

    max_cosine_dists = [round(i * 0.05, 2) for i in range(1, 20)]
    max_iou_dists = [round(i * 0.05, 2) for i in range(1, 20)]
    max_ages = list(range(1, 11))
    n_inits = list(range(1, 11))

    with open(detection_results_path, encoding="utf-8") as f:
        preds = json.load(f)

    preds = sorted(preds, key=lambda det: det["image_id"])

    # todo go over the sequence see if no preds are made
    preds_2d = []
    last_image = []
    image_id = preds[0]["image_id"]
    for pred in preds:
        new_image_id = pred["image_id"]
        if new_image_id != image_id:
            preds_2d.append((image_id, last_image))
            last_image = [pred]
            image_id = new_image_id
            continue
        last_image.append(pred)

    preds_2d.append((image_id, last_image))

    for max_cosine_dist in max_cosine_dists:
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_dist, None
        )
        for max_iou_dist in max_iou_dists:
            for max_age in max_ages:
                for n_init in n_inits:
                    result = track(
                        metric,
                        max_iou_dist,
                        max_age,
                        n_init,
                        preds_2d,
                        model,
                        dataset_path,
                        train_mean_patch_dim,
                        final_patch_dim,
                        detector_image_dim,
                    )
                    with open(
                        f"/Users/reza/Career/DMLab/AR_TRACKING/Results/track/deepsort/val/{result_name}/max_cosine_dist_{max_cosine_dist}_max_iou_dist_{max_age}_max_age_{max_age}_n_init_{n_init}.txt",
                        "w",
                        encoding="utf-8",
                    ) as out:
                        for line in result:
                            line_str = ", ".join([str(l) for l in line])
                            out.write(line_str + "\n")
                    print(
                        f"{result_name}/max_cosine_dist_{max_cosine_dist}_max_iou_dist_{max_age}_max_age_{max_age}_n_init_{n_init}.txt"
                    )


def track(
    metric,
    max_iou_dist,
    max_age,
    n_init,
    preds,
    encoder,
    dataset_path,
    train_mean_patch_dim,
    final_patch_dim,
    detector_image_dim,
):
    tracker = Tracker(metric, max_iou_dist, max_age, n_init)
    result = []
    for i, frame in enumerate(preds):
        image_id, predictions = frame
        image = (
            cv.imread(
                os.path.join(dataset_path, image_id + ".jpg"), cv.IMREAD_GRAYSCALE
            )
            / 255.0
        )
        boxes = [[*pred["bbox"], pred["score"]] for pred in predictions]
        feats = []
        final_boxes = []
        for box in boxes:
            x1, y1, w, h, _ = box
            patch = get_patch(
                image,
                (x1, y1, w, h),
                train_mean_patch_dim,
                final_patch_dim,
                detector_image_dim,
            )
            feat = encoder(patch.to(device))
            feats.append(feat.squeeze())
            final_boxes.append(box)
        dets = [
            Detection(b[:4], b[4], feats[i].detach().numpy())
            for i, b in enumerate(final_boxes)
        ]

        tracker.predict()
        tracker.update(dets)
        # fig, ax = plt.subplots(1, figsize=(10, 10))
        # ax.axis("off")
        # ax.imshow(image, cmap="gray")
        # print(tracker.tracks)
        for t in tracker.tracks:
            if not t.is_confirmed():
                continue
            [x1, y1, w, h] = t.to_tlwh()
            result.append([i + 1, t.track_id, x1, y1, w, h, t.confidence, -1, -1, -1])
            # print(x1, y1, w, h)
            # rect = patches.Rectangle(
            #     (x1, y1), w, h, linewidth=2, edgecolor="red", facecolor="none"
            # )
            # ax.text(
            #     x1,
            #     y1,
            #     str(t.track_id),
            #     color="white",
            #     fontsize=10,
            #     ha="center",
            #     va="center",
            #     bbox=dict(facecolor="red", alpha=0.5, lw=0)
            # )
            # ax.add_patch(rect)
        plt.show()
    return result


def get_patch(image, box, train_mean_patch_dim, final_patch_dim, detector_image_dim):
    if image.shape != (detector_image_dim, detector_image_dim):
        image = cv.resize(
            image,
            (detector_image_dim, detector_image_dim),
            interpolation=cv.INTER_LINEAR,
        )
    half_patch_dim = train_mean_patch_dim // 2
    (x1, y1, w, h) = box
    x = int(x1 + w / 2)
    y = int(y1 + h / 2)
    x_min, x_max = max(0, x - half_patch_dim), min(
        detector_image_dim, x + half_patch_dim
    )
    y_min, y_max = max(0, y - half_patch_dim), min(
        detector_image_dim, y + half_patch_dim
    )
    patch = image[
        y_min:y_max,
        x_min:x_max,
    ]
    pad_height = max(0, half_patch_dim * 2 - patch.shape[0])
    pad_width = max(0, half_patch_dim * 2 - patch.shape[1])

    if pad_width > 0:
        if x > detector_image_dim // 2:
            patch = np.pad(
                patch,
                (
                    (0, 0),
                    (0, pad_width),
                ),
                mode="constant",
                constant_values=0,
            )
        else:
            patch = np.pad(
                patch,
                (
                    (0, 0),
                    (pad_width, 0),
                ),
                mode="constant",
                constant_values=0,
            )
    if pad_height > 0:
        if y > detector_image_dim // 2:
            patch = np.pad(
                patch,
                (
                    (0, pad_height),
                    (0, 0),
                ),
                mode="constant",
                constant_values=0,
            )
        else:
            patch = np.pad(
                patch,
                (
                    (pad_height, 0),
                    (0, 0),
                ),
                mode="constant",
                constant_values=0,
            )
    if patch.shape != (final_patch_dim, final_patch_dim):
        patch = cv.resize(patch, (final_patch_dim, final_patch_dim))
    patch = np.asarray(patch, dtype=np.float32)
    patch = np.expand_dims(patch, 0)
    patch = np.expand_dims(patch, 0)
    return torch.tensor(patch, dtype=torch.float32)


if __name__ == "__main__":
    main()
