import os
import glob
import random
import shutil
from tqdm import tqdm

# Define source and destination paths
SOURCE_DIR = "/scratch/rksing18/stpp/waymo/datasets/waymo_pred_downsampled/frame_rate_2"
DEST_DIR = "/scratch/rksing18/stpp/waymo/sampled_datasets/waymo_pred_downsampled/frame_rate_2"

# Delete DEST_DIR if it exists
if os.path.exists(DEST_DIR):
    shutil.rmtree(DEST_DIR)

# Ensure destination directory exists
os.makedirs(DEST_DIR)

# Subfolders to process
subfolders = ["train", "val", "test"]
file_counts = {"train": 500, "val": 150, "test": 150}

# Process each subfolder
for subfolder in subfolders:
    src_folder = os.path.join(SOURCE_DIR, "label", subfolder)
    dest_folder = os.path.join(DEST_DIR, "label", subfolder)
    os.makedirs(dest_folder, exist_ok=True)

    # Get all scene files in the subfolder
    scene_files = glob.glob(os.path.join(src_folder, "scene-*.txt"))
    selected_files = random.sample(scene_files, file_counts[subfolder])

    for scene_file in tqdm(selected_files, desc=f"Processing {subfolder}"):
        # Copy scene file
        shutil.copy(scene_file, dest_folder)

        # Extract scene ID
        scene_id = os.path.basename(scene_file).split("-")[1].split(".")[0]

        # Copy corresponding map files
        map_files = [
            f"meta_scene-{scene_id}.txt",
            f"scene-{scene_id}.png",
            f"vis_scene-{scene_id}.png",
        ]
        for map_file in map_files:
            src_map_file = os.path.join(SOURCE_DIR, "map_0.1", map_file)
            dest_map_folder = os.path.join(DEST_DIR, "map_0.1")
            os.makedirs(dest_map_folder, exist_ok=True)
            shutil.copy(src_map_file, dest_map_folder)
