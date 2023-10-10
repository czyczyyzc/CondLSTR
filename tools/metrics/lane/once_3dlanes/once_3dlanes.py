import os
import sys
import cv2
import json
import pickle
import argparse
import subprocess
import numpy as np
from tqdm import tqdm


def main():
    args = parse_args()
    print('#############################################################')
    print(args.pred_dir)
    print(args.anno_dir)

    if args.translate:
        with open(os.path.join(args.pred_dir, 'results.pkl'), 'rb') as f:
            results_list = pickle.load(f)

        for result in tqdm(results_list):
            ann_path = os.path.join(args.anno_dir, 'test',
                                    result['image_name'].replace(',', '/').replace('.jpg', '.json'))
            img_path = os.path.join(args.anno_dir, 'data',
                                    result['image_name'].replace(',', '/'))
            res_path = os.path.join(args.pred_dir, 'test',
                                    result['image_name'].replace(',', '/').replace('.jpg', '.json'))

            if not (os.path.exists(ann_path) and os.path.exists(img_path)):
                print(ann_path)
                print(img_path)
                continue

            cam_pitch = 0.5 / 180 * np.pi
            cam_height = 1.5
            cam_extrinsics = np.array([[np.cos(cam_pitch), 0, -np.sin(cam_pitch), 0],
                                       [0, 1, 0, 0],
                                       [np.sin(cam_pitch), 0, np.cos(cam_pitch), cam_height],
                                       [0, 0, 0, 1]], dtype=float)
            R_vg = np.array([[0, 1, 0],
                             [-1, 0, 0],
                             [0, 0, 1]], dtype=float)
            R_gc = np.array([[1, 0, 0],
                             [0, 0, 1],
                             [0, -1, 0]], dtype=float)
            cam_extrinsics[:3, :3] = np.matmul(np.matmul(
                np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
                R_vg), R_gc)
            cam_extrinsics[0:2, 3] = 0.0

            lanelines_pred = result['3d_res']
            lanelines_conf = result['score']

            lane_lines = []
            for k in range(len(lanelines_pred)):
                # if np.max(lanelines_prob[k]) < 0.5: #TODO
                #     continue
                lane = np.array(lanelines_pred[k])[:, :3]
                lane = np.flip(lane, axis=0)
                lane = lane.T
                lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
                lane = np.matmul(np.linalg.inv(cam_extrinsics), lane)
                lane = lane[0:3, :].T
                # x = lane[:,0]
                # y = lane[:,2] + 1.8
                # z = lane[:,1]
                # lane = np.stack([x,y,z],0).T
                # lane_lines.append(lane.tolist())
                score = float(lanelines_conf[k])
                lane_lines.append({
                    'points': lane.tolist(),
                    'score': score,
                })
            result_dict = {
                "lanes": lane_lines,
            }

            os.makedirs(os.path.dirname(res_path), exist_ok=True)
            with open(res_path, 'w') as f:
                json.dump(result_dict, f)

    # python eval.py --gt_path gt_dir --pred_path pred_dir
    current_dir = os.path.split(os.path.realpath(__file__))[0]
    subprocess.run([sys.executable, os.path.join(current_dir, 'eval.py'),
                    '--cfg_file', os.path.join(current_dir, 'cfg/eval.json'),
                    '--gt_path', os.path.join(args.anno_dir, 'test'),
                    '--pred_path', os.path.join(args.pred_dir, 'test')])


def parse_args():
    parser = argparse.ArgumentParser(description="Measure ONCE_3DLanes's metric")
    parser.add_argument(
        "--pred_dir",
        default='/data/gpfs/projects/punim1962/project/HDMapNet/output/test_bev_cond_lstr_attr_once_proj_3d_swin_2_laetst',
        help="Path to directory containing the predicted lanes"
    )
    parser.add_argument(
        "--anno_dir",
        default='/data/gpfs/projects/punim1962/datasets/once_3dlanes/',
        help="Path to directory containing the annotated lanes"
    )
    parser.add_argument("--top_view_region",
                        type=list,
                        default=[[-10, 103], [10, 103], [-10, 3], [10, 3]],
                        help="top_view_region")
    parser.add_argument("--list",
                        nargs='+',
                        default=['lane3d_1000/validation'],
                        # default=['lane3d_1000/test/' + s for s in ['up_down_case', 'curve_case', 'extreme_weather_case', 'intersection_case', 'merge_split_case', 'night_case']],
                        help="Path to txt file containing the list of files"
    )
    parser.add_argument('--translate', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    main()
