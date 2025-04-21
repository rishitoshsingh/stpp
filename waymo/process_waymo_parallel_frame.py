import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from waymo_open_dataset.protos import scenario_pb2

from waymo_types import object_type, lane_type, road_line_type, road_edge_type, signal_state, polyline_type

def decode_tracks_from_proto(tracks, tracks_to_predict, frame_rate=10):
    tracks_to_predict_ids = [x.track_index for x in tracks_to_predict]
    data = None
    for cur_data in tracks:
        cur_traj = [np.array([
            -1.0, -1.0, -1.0, -1.0,
            x.velocity_x, x.velocity_y,
            # x.width, x.length, x.height,
            x.width, x.height, x.length,  # from process_nuscenes.py line 127
            # x.center_x, x.center_y, x.center_z,
            x.center_x, x.center_z, x.center_y, # from process_nuscenes.py line 130
            x.heading, 1.0 if cur_data.id in tracks_to_predict_ids else 0.0
        ], dtype=np.float32) for i, x in enumerate(cur_data.states) if x.valid and i % int(10/frame_rate) == 0]
        try:
            cur_traj = np.stack(cur_traj, axis=0)
        except:
            continue
        _data = np.hstack(( np.arange(1, cur_traj.shape[0] + 1, dtype=np.float32).reshape(-1, 1), 
                           np.full((cur_traj.shape[0], 1), cur_data.id, dtype=np.float32),
                           np.array([object_type.get(cur_data.object_type)]*cur_traj.shape[0]).reshape(-1,1), 
                           cur_traj))
        if data is None:
            data = _data
        else:
            data = np.vstack((data, _data))
    data = data[np.lexsort((data[:, 1].astype(float), data[:, 0].astype(float)))]
    return data

# No change to decode_map_features_from_proto and plot_map_features...

def process_single_tfrecord(file_path, split, DATAOUT, map_version, frame_rate=10):
    try:
        raw_dataset = tf.data.TFRecordDataset(file_path, compression_type="")
        for serialized_scenario in raw_dataset:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(serialized_scenario.numpy())
            scenario_id = scenario.scenario_id

            label_path = f'{DATAOUT}/label/{split}/scene-{scenario_id}.txt'
            if os.path.exists(label_path):
                continue

            track_infos = decode_tracks_from_proto(scenario.tracks, scenario.tracks_to_predict, frame_rate=frame_rate)
            map_infos = decode_map_features_from_proto(scenario.map_features)
            meta, vis_canvas, dark_canvas = plot_map_features(map_infos)

            np.savetxt(f'{DATAOUT}/map_{map_version}/meta_scene-{scenario_id}.txt', meta, fmt='%.2f')
            cv2.imwrite(f'{DATAOUT}/map_{map_version}/scene-{scenario_id}.png', dark_canvas)
            cv2.imwrite(f'{DATAOUT}/map_{map_version}/vis_scene{scenario_id}.png', vis_canvas)

            np.savetxt(label_path, track_infos, fmt='%s')
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main(data_root, data_out, map_version="0.1", num_workers=os.cpu_count(), frame_rate=10):
    splits = ['train', 'val', 'test']

    for split in splits:
        os.makedirs(f'{data_out}/label/{split}', exist_ok=True)
        os.makedirs(f'{data_out}/map_{map_version}', exist_ok=True)

        if split == 'train':
            src_files = glob.glob(f"{data_root}/training/*.tfrecord*")
        elif split == 'val':
            src_files = glob.glob(f"{data_root}/validation/*.tfrecord*")
        elif split == 'test':
            src_files = glob.glob(f"{data_root}/testing/*.tfrecord*")
        src_files.sort()

        print(f"Processing split '{split}' with {len(src_files)} files using {num_workers} workers.")

        process_func = partial(
            process_single_tfrecord,
            split=split,
            DATAOUT=data_out,
            map_version=map_version,
            frame_rate=frame_rate
        )

        with Pool(num_workers) as pool:
            list(tqdm(pool.imap(process_func, src_files), total=len(src_files)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help="Path to the original Waymo Motion dataset")
    parser.add_argument('--data_out', default='datasets/waymo_pred/', help="Output path for processed data")
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help="Number of parallel workers")
    parser.add_argument('--frame_rate', type=int, default=10, help="Target frame rate for downsampling (e.g., 2, 5, 10)")
    args = parser.parse_args()

    main(args.data_root, args.data_out, num_workers=args.num_workers, frame_rate=args.frame_rate)
