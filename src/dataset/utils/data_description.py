import os
import sys
from tqdm import trange
from datetime import datetime, timedelta


def get_time(file_name, extension):
    file_name = file_name[: -(len(extension) + 1)]
    t = datetime.strptime(file_name, "%Y-%m-%dT%H:%M:%S.%f").replace(
        minute=0, second=0, microsecond=0
    )
    return t


def main():
    labels_path = sys.argv[1]
    label_names = sorted(os.listdir(labels_path))
    first_name = label_names[0]
    extension = first_name.split(".")[-1]
    interval = timedelta(hours=4)
    sequences = []
    current_sequence = [first_name]
    for i in trange(1, len(label_names)):
        current_label_name = label_names[i]
        previous_label_name = label_names[i - 1]
        current_time = get_time(current_label_name, extension)
        previous_time = get_time(previous_label_name, extension)
        if previous_time + interval == current_time:
            current_sequence.append(current_label_name)
        else:
            sequences.append(current_sequence)
            current_sequence = [current_label_name]
    if len(current_sequence) != 0:
        sequences.append(current_sequence)
    with open("file_sequences_results.txt", "w", encoding="utf-8") as result:
        for sequence in sequences:
            result.write(
                f"length: {len(sequence)} first: {sequence[0]} last: {sequence[-1]}\n"
            )
            for time in sequence:
                result.write(f"{time}\n")
            result.write("=" * 60)


if __name__ == "__main__":
    main()
