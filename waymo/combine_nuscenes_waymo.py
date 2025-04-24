import os
import shutil
import argparse

def combine_datasets(nuscenes_data, waymo_data, merged_data_path):
    # Delete NEW_MERGED_DATA_PATH if it exists
    if os.path.exists(merged_data_path):
        shutil.rmtree(merged_data_path)
        print(f"Deleted existing directory: {merged_data_path}")
    
    # Create NEW_MERGED_DATA_PATH
    os.makedirs(merged_data_path)
    print(f"Created new directory: {merged_data_path}")
    
    # Define subdirectories to copy
    subdirs = ["label/train", "label/val", "label/test", "map_0.1"]
    
    # Copy contents from NUSCENES_DATA
    for subdir in subdirs:
        src = os.path.join(nuscenes_data, subdir)
        dest = os.path.join(merged_data_path, subdir)
        if os.path.exists(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
            print(f"Copied {src} to {dest}")
    
    # Copy contents from WAYMO_DATA
    for subdir in subdirs:
        src = os.path.join(waymo_data, subdir)
        dest = os.path.join(merged_data_path, subdir)
        if os.path.exists(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
            print(f"Copied {src} to {dest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine NuScenes and Waymo datasets into a single directory.")
    parser.add_argument("--nuscenes_data", help="Path to the NuScenes processed data", default="/scratch/rksing18/stpp/AgentFormer/datasets/nuscenes_pred")
    parser.add_argument("--waymo_data", help="Path to the Waymo processed data", default="/scratch/rksing18/stpp/waymo/sampled_processed")
    parser.add_argument("--merged_data_path", help="Path to the new merged dataset directory", default="/scratch/rksing18/stpp/AgentFormer/datasets/merged_nuscenes_waymo")
    
    args = parser.parse_args()
    
    combine_datasets(args.nuscenes_data, args.waymo_data, args.merged_data_path)
