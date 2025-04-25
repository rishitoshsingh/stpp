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
            -1.0, -1.0, -1.0, -1.0, -1.0,
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
def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir

def decode_map_features_from_proto(map_features):
    map_infos = {
        'lane': [],
        'road_line': [],
        'road_edge': [],
        'stop_sign': [],
        'crosswalk': [],
        'speed_bump': []
    }
    polylines = []

    point_cnt = 0
    for i, cur_data in enumerate(map_features):
        cur_info = {'id': cur_data.id}

        if cur_data.HasField("lane"):
            cur_info['speed_limit_mph'] = cur_data.lane.speed_limit_mph
            cur_info['type'] = lane_type[cur_data.lane.type]

            cur_info['interpolating'] = cur_data.lane.interpolating
            cur_info['entry_lanes'] = list(cur_data.lane.entry_lanes)
            cur_info['exit_lanes'] = list(cur_data.lane.exit_lanes)

            cur_info['left_boundary'] = [{
                    'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                    'feature_id': x.boundary_feature_id,
                    'boundary_type': x.boundary_type
                } for x in cur_data.lane.left_boundaries
            ]
            cur_info['right_boundary'] = [{
                    'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                    'feature_id': x.boundary_feature_id,
                    'boundary_type': road_line_type[x.boundary_type]
                } for x in cur_data.lane.right_boundaries
            ]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.lane.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['lane'].append(cur_info)

        elif cur_data.HasField("road_line"):
            cur_info['type'] = road_line_type[cur_data.road_line.type]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_line.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_line'].append(cur_info)

        elif cur_data.HasField("road_edge"):
            cur_info['type'] = road_edge_type[cur_data.road_edge.type]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_edge.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_edge'].append(cur_info)

        elif cur_data.HasField("stop_sign"):
            cur_info['lane_ids'] = list(cur_data.stop_sign.lane)
            point = cur_data.stop_sign.position
            cur_info['position'] = np.array([point.x, point.y, point.z])

            global_type = polyline_type['TYPE_STOP_SIGN']
            cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 7)

            map_infos['stop_sign'].append(cur_info)

        elif cur_data.HasField("crosswalk"):
            global_type = polyline_type['TYPE_CROSSWALK']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.crosswalk.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['crosswalk'].append(cur_info)

        elif cur_data.HasField("speed_bump"):
            global_type = polyline_type['TYPE_SPEED_BUMP']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.speed_bump.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['speed_bump'].append(cur_info)

        else:
            continue

        polylines.append(cur_polyline)
        cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 7), dtype=np.float32)
        print('Empty polylines: ')
    map_infos['all_polylines'] = polylines
    return map_infos

def plot_map_features(map_infos, save_path="map.png", meta_path="meta.txt"):
    color_map = {
        'lane': [219, 225, 200],
        # 'road_line': [40, 40, 40],
        'road_edge': [0, 0, 0],
        'stop_sign': [0, 0, 255],
        'crosswalk': [116, 116, 116],
        'speed_bump': [53, 209, 243],
        'rest': [243, 240, 255]
    }

    dark_color_map = {
        'lane': [246, 9, 0],
        'road_line': [219, 225, 200],
        # 'road_line': [20, 20, 20],
        # 'road_edge': [179, 185, 160],
        'road_edge': [20, 20, 20],
        'stop_sign': [0, 0, 255],
        'crosswalk': [116, 116, 116],
        'speed_bump': [53, 209, 243],
        'rest': [20, 20, 20]
    }

    thickness_map = {
        'lane': 13,
        'road_line': 3,
        'road_edge': 2,
        'stop_sign': 4,
        'crosswalk': 5,
        'speed_bump': 4,
    }

    all_polylines = map_infos.get("all_polylines", None)
    if all_polylines is None or len(all_polylines) == 0:
        print("No polylines found.")
        return

    # Calculate bounds with margin
    margin = 75
    scale = 2
    x = all_polylines[:, 0]
    y = all_polylines[:, 1]
    x_min, x_max = np.round(x.min() - margin), np.round(x.max() + margin)
    y_min, y_max = np.round(y.min() - margin), np.round(y.max() + margin)
    x_size = int(np.round((x_max - x_min) * scale))
    y_size = int(np.round((y_max - y_min) * scale))

    def draw_map(canvas, cmap):
        for feature_type, color in cmap.items():
            if feature_type == 'rest':
                continue
            thickness = thickness_map.get(feature_type, 1)
            for item in map_infos[feature_type]:
                start, end = item['polyline_index']
                polyline = all_polylines[start:end]
                pts = np.round((polyline[:, :2] - [x_min, y_min]) * scale).astype(int)
                for i in range(len(pts) - 1):
                    cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), color=color, thickness=thickness)
        return canvas
    

    # Generate visual canvas
    vis_canvas = np.ones((y_size, x_size, 3), dtype=np.uint8) * np.array(color_map['rest'], dtype=np.uint8)
    vis_canvas = draw_map(vis_canvas, color_map)

    # Generate dark canvas
    dark_canvas = np.ones((y_size, x_size, 3), dtype=np.uint8) * np.array(dark_color_map['rest'], dtype=np.uint8)
    dark_canvas = draw_map(dark_canvas, dark_color_map)

    # Save metadata
    meta = np.array([x_min, y_min, scale])
    return meta, vis_canvas, dark_canvas


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
            cv2.imwrite(f'{DATAOUT}/map_{map_version}/vis_scene-{scenario_id}.png', vis_canvas)

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
        src_files = src_files[:50]

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
    parser.add_argument('--data_root', default='/scratch/rksing18/datasets/waymo/scenario', help="Path to the original Waymo Motion dataset")
    parser.add_argument('--data_out', default='datasets/waymo_pred_downsampled/', help="Output path for processed data")
    # parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help="Number of parallel workers")
    parser.add_argument('--num_workers', type=int, default=10, help="Number of parallel workers")
    parser.add_argument('--frame_rate', type=int, default=2, help="Target frame rate for downsampling (e.g., 2, 5, 10)")
    args = parser.parse_args()
    args.data_out = args.data_out + f"frame_rate_{args.frame_rate}/"
    main(args.data_root, args.data_out, num_workers=args.num_workers, frame_rate=args.frame_rate)
