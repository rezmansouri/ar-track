import os
import sys
import pandas as pd
# from pprint import pprint


def main():
    results_path = sys.argv[1]
    iou_threshs = [round(i * 0.05, 2) for i in range(1, 20)]
    conf_threshs = [round(i * 0.05, 2) for i in range(1, 20)]
    results = []
    for iou in iou_threshs:
        for conf in conf_threshs:
            csv_path = os.path.join(results_path, f"conf-{conf}-iou{iou}", "maps.csv")
            csv = pd.read_csv(csv_path)
            map5095 = csv["map50-95"][0]
            results.append((conf, iou, map5095))

    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
    print("scale:", results_path[-1])
    # pprint(results_sorted)
    print(results_sorted[0])


if __name__ == "__main__":
    main()
