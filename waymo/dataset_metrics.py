import os
import numpy as np
from tqdm import tqdm

def calculate_sums(data_dir):
    subdirs = ['train', 'val', 'test']
    total_sums = {}

    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, 'label', subdir)
        total_sum = 0

        if not os.path.exists(subdir_path):
            print(f"Directory {subdir_path} does not exist. Skipping...")
            continue

        # List all .txt files in the subdirectory
        txt_files = [f for f in os.listdir(subdir_path) if f.endswith('.txt')]

        # Use tqdm to display progress
        for file_name in tqdm(txt_files, desc=f"Processing {subdir}"):
            file_path = os.path.join(subdir_path, file_name)
            # Load the file as a numpy array
            data = np.genfromtxt(file_path, delimiter=' ', dtype=str)
            total_sum += np.sum(data[:, -1].astype(float))

        total_sums[subdir] = total_sum

    return total_sums


if __name__ == "__main__":
    # DATA_DIR = "/path/to/your/dataset"  # Replace with the actual dataset path
    DATA_DIR = "sampled_processed/"  # Replace with the actual dataset path
    DATA_DIR = "/scratch/rksing18/stpp/AgentFormer/datasets/merged_nuscenes_waymo/"  # Replace with the actual dataset path

    sums = calculate_sums(DATA_DIR)
    for subdir, total_sum in sums.items():
        print(f"Total sum for {subdir}: {total_sum}")