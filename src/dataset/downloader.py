import os
import sys
import pandas as pd
import astropy.units as u
from sunpy.map import Map
from tqdm import tqdm, trange
from datetime import datetime
from astropy.coordinates import SkyCoord
from sunpy.net import Fido, hek, attrs as a


def main():
    start_date, end_date, cadance = sys.argv[1:]
    time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    data_path = os.path.join(".", f"{time}-S{start_date}E{end_date}C{cadance}")
    images_path = os.path.join(data_path, "images")
    labels_path = os.path.join(data_path, "labels")
    os.mkdir(data_path)
    os.mkdir(images_path)
    os.mkdir(labels_path)
    # cadance 4 is preferred
    results = Fido.search(
        a.Time(start_date, end_date),
        a.Instrument.hmi,
        a.Physobs.los_magnetic_field,
        a.Sample(int(cadance) * u.hour),
    )
    n_found = sum(len(block) for block in results)
    files = Fido.fetch(results, path=images_path)
    if n_found != len(files):
        print(f"{n_found} files found, {len(files)} downloaded")
    hek_client = hek.HEKClient()

    ars = []
    times = []
    print("Finding ARs")
    for file in tqdm(files):
        if "err" in file:
            os.remove(file)
            continue
        magnetogram = Map(file)
        observer = magnetogram.observer_coordinate
        # os.rename(file, os.path.join(images_path, f"{str(magnetogram.date)}.fits"))
        times.append(str(magnetogram.date))
        active_regions = get_noaa_active_regions(hek_client, magnetogram.date)
        rectangles = dict()
        for region in active_regions:
            if region["ar_noaanum"] is None:
                continue
            hpc_coords = [
                tuple(map(float, point.split()))
                for point in region["hpc_bbox"]
                .replace("POLYGON((", "")
                .replace("))", "")
                .split(",")
            ]
            hpc_vertices = SkyCoord(
                [x[0] for x in hpc_coords] * u.arcsec,
                [x[1] for x in hpc_coords] * u.arcsec,
                frame="helioprojective",
                observer=observer,
            )
            pixel_coords = magnetogram.world_to_pixel(hpc_vertices)
            min_x, min_y = int(pixel_coords.x.min().value), int(
                pixel_coords.y.min().value
            )
            max_x, max_y = int(pixel_coords.x.max().value), int(
                pixel_coords.y.max().value
            )
            if region["ar_noaanum"] in rectangles:
                rectangles[region["ar_noaanum"]].append(
                    (min_x, min_y, max_x - min_x, max_y - min_y)
                )
            else:
                rectangles[region["ar_noaanum"]] = [
                    (min_x, min_y, max_x - min_x, max_y - min_y)
                ]
        rectangles = {
            ar_num: sorted(
                rectangles[ar_num], key=lambda bbox: bbox[2] * bbox[3], reverse=True
            )[0]
            for ar_num in rectangles
        }
        ars.append(rectangles)
    stills = get_still_arnos(ars)
    print("Writing labels")
    for i in trange(len(ars)):
        ar = ars[i]
        timee = times[i]
        label_dict = {
            "ar_noaanum": [],
            "min_x": [],
            "min_y": [],
            "width": [],
            "height": [],
        }
        for ar_num in ar:
            if ar_num in stills:
                continue
            label_dict["ar_noaanum"].append(ar_num)
            min_x, min_y, width, height = ar[ar_num]
            label_dict["min_x"].append(min_x)
            label_dict["min_y"].append(min_y)
            label_dict["width"].append(width)
            label_dict["height"].append(height)
        if len(label_dict["ar_noaanum"]) == 0:
            continue
        df = pd.DataFrame(label_dict)
        df.to_csv(os.path.join(labels_path, f"{timee}.csv", index=False))


def get_noaa_active_regions(client, magnetogram_time):
    active_regions = client.search(
        a.Time(magnetogram_time, magnetogram_time + 1 * u.minute),
        hek.attrs.EventType("AR"),
    )
    filtered = [ar for ar in active_regions if ar.get("obs_instrument") == "HMI"]
    return filtered


def longest_consecutive_subarray(nums):
    longest_streak = 0
    current_streak = 1

    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            current_streak += 1
        else:
            longest_streak = (
                longest_streak if longest_streak > current_streak else current_streak
            )
            current_streak = 1
    longest_streak = (
        longest_streak if longest_streak > current_streak else current_streak
    )
    return longest_streak


def get_still_arnos(all_ars):
    counts = dict()
    n_thresh = 5
    for i, instance in enumerate(all_ars):
        if i == len(all_ars) - 1:
            break
        other_instance = all_ars[i + 1]
        for ar_no in instance:
            min_x_a, min_y_a, width_a, height_a = instance[ar_no]
            if ar_no not in other_instance:
                continue
            min_x_b, min_y_b, width_b, height_b = other_instance[ar_no]
            if (
                min_x_a == min_x_b
                and min_y_a == min_y_b
                and width_a == width_b
                and height_a == height_b
            ):
                if ar_no in counts:
                    counts[ar_no].append(i + 1)
                else:
                    counts[ar_no] = [i, i + 1]
    counts = {ar_no: list(c) for ar_no, c in counts.items()}
    new_counts = {ar_no: [] for ar_no in counts}
    for ar_no in counts:
        for i in range(len(counts[ar_no]) - 1):
            if counts[ar_no][i] == counts[ar_no][i + 1] - 1:
                new_counts[ar_no].append(counts[ar_no][i])
    new_new_counts = {
        ar_no: longest_consecutive_subarray(new_counts[ar_no]) for ar_no in new_counts
    }
    stills = [ar_no for ar_no in new_new_counts if new_new_counts[ar_no] >= n_thresh]
    return stills


if __name__ == "__main__":
    main()
