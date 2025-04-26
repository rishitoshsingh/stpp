import sys, os
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import argparse
import shutil

LABEL_MAPPER = {
    'Skater': 'Cyclist',
    'Bus': 'Bus',
    'Car': 'Car',
    'Cart': 'Van',
    'Biker': 'Cyclist',
    'Pedestrian': 'Pedestrian',
}

def read_stanford_data(data_path):
    header = ["track", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
    data = pd.read_csv(data_path, sep=' ', encoding='utf-8', header=None, names=header)
    # finding coordinate of agent
    data['x'] = (data["xmin"].astype("int") + data["xmax"].astype("int") ) // 2
    data['y'] = (data["ymin"].astype("int") + data["ymax"].astype("int") ) // 2
    data.drop(columns=["xmin", "ymin", "xmax", "ymax"], inplace=True)

    # dropping lost, occuluded or generated trajectories
    idx = ((data["lost"] == 1 ) | ( data["occluded"] == 1 ))# | (data["generated"] == 1 ))
    data = data[~idx]
    data.drop(columns=["lost", "occluded", "generated"], inplace=True)

    # Map data["label"] using LABEL_MAPPER
    data["label"] = data["label"].map(LABEL_MAPPER)

    cols = ["frame", "track", "label", "x", "y"]
    data = data[cols]
    data.sort_values(by=["frame", "track"], inplace=True, kind="mergesort")

    # Normalize columns x and y
    new_min = 1
    new_max = 20
    data['x'] = (data['x'] - data['x'].min()) / (data['x'].max() - data['x'].min()) * (new_max - new_min) + new_min
    data['y'] = (data['y'] - data['y'].min()) / (data['y'].max() - data['y'].min()) * (new_max - new_min) + new_min

    # insert don't care columns 
    target_cols = ["frame", "track", "label", "dont_care1", "dont_care2", "dont_care3", "dont_care4", "dont_care5", "dont_care6", "dont_care7", "dont_care8", "dont_care9", "dont_care10", "x", "dont_care11", "y", "dont_care12"]
    for col in target_cols:
        if col.startswith("dont_care"):
            data[col] = -1
    
    data = data[target_cols]
    # Change columns datatype from int to float
    data = data.astype({col: 'float' for col in data.select_dtypes(include='int').columns})
    return data
    

def process_stanford_dataset(data_root, eth_ecy_dir):
    """
    Process the Stanford dataset and save the processed data to the output directory.
    
    Args:
        data_root (str): Path to the root directory of the Stanford dataset.
        output_dir (str): Path to the output directory where processed data will be saved.
        split (str): The split of the dataset to process ('train' or 'val').
    """
    
    # Copy the eth_ecy_dir to a new directory ending with '_stanford'
    eth_ecy_stanford_dir = eth_ecy_dir.rstrip('/') + '_stanford'
    if os.path.exists(eth_ecy_stanford_dir):
        shutil.rmtree(eth_ecy_stanford_dir)
    print("copying original dataset to {}".format(eth_ecy_stanford_dir))
    shutil.copytree(eth_ecy_dir, eth_ecy_stanford_dir)

    # Rename all subdirectories in eth_ecy_stanford_dir by appending '_stanford' to their names
    for subdir in os.listdir(eth_ecy_stanford_dir):
        subdir_path = os.path.join(eth_ecy_stanford_dir, subdir)
        if os.path.isdir(subdir_path):
            new_subdir_path = os.path.join(eth_ecy_stanford_dir, subdir + '_stanford')
            os.rename(subdir_path, new_subdir_path)
    print("copying done")

    # Find all directories in eth_ucy_stanford directory
    eth_ucy_stanford_subdirs = [os.path.join(eth_ecy_stanford_dir, d) for d in os.listdir(eth_ecy_stanford_dir) if os.path.isdir(os.path.join(eth_ecy_stanford_dir, d))]

    scenes_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    for scene_dir in tqdm(scenes_dirs, desc="Processing scenes"):
        video_dirs = [os.path.join(scene_dir, d) for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))]
        random_video_dir = np.random.choice(video_dirs)
        ann_data_path = os.path.join(random_video_dir, 'annotations.txt')
        data = read_stanford_data(ann_data_path)

        # data_file_name = f"stanford_{scene_dir.split('/')[-1]}_{random_video_dir.split('/')[-1]}.txt"
        data_file_name = f"stanford_{scene_dir.split('/')[-1]}.txt"
        for subdir in eth_ucy_stanford_subdirs:
            data.to_csv(os.path.join(subdir, data_file_name), sep=' ', header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Stanford dataset')
    parser.add_argument('--data_root', type=str, default='StanfordDroneDataset', help='Path to the Stanford dataset')
    parser.add_argument('--eth_ucy_dir', type=str, default='../AgentFormer/datasets/eth_ucy', help='original eth_ucy data directory')
    args = parser.parse_args()

    process_stanford_dataset(args.data_root, args.eth_ucy_dir)

