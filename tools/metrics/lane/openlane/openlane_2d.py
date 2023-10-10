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

        os.makedirs(os.path.join(args.pred_dir, 'validation'), exist_ok=True)
        with open(os.path.join(args.pred_dir, 'validation/test_list.txt'), 'w') as ff:
            for result in tqdm(results_list):
                ann_path = os.path.join(args.anno_dir, 'lane3d_1000/validation',
                                        result['image_name'].replace(',', '/').replace('.jpg', '.json'))
                img_path = os.path.join(args.anno_dir, 'images/validation',
                                        result['image_name'].replace(',', '/'))
                res_path = os.path.join(args.pred_dir, 'validation',
                                        result['image_name'].replace(',', '/').replace('.jpg', '.json'))
                if not (os.path.exists(ann_path) and os.path.exists(img_path)):
                    print(ann_path)
                    print(img_path)
                    continue
                lanelines_pred = [np.array(x, dtype=np.float32).reshape(-1, 2).T.tolist() for x in result['points']]
                lanelines_cate = result['attris']

                file_path = '/'.join(img_path.split('/')[-3:])
                result_dict = {
                    "file_path": file_path,                      # < str > -- image path
                    "lane_lines": [
                        {
                            "uv": uv,                            # < float > [3, n] - - x, y, z coordinates of sampled points in vehicle coordinate
                            "category": category,                # < int > -- lane category
                        }
                        for uv, category in zip(lanelines_pred, lanelines_cate)
                    ]
                }

                os.makedirs(os.path.dirname(res_path), exist_ok=True)
                with open(res_path, 'w') as f:
                    json.dump(result_dict, f)

                ff.write(file_path + '\n')

    # python lane3d/eval_3D_lane.py --dataset_dir $dataset_dir --pred_dir $pred_dir --test_list $test_list

    current_dir = os.path.split(os.path.realpath(__file__))[0]

    if not args.list[0]:
        print(os.path.join(args.anno_dir, 'lane3d_1000/validation'))
        subprocess.call([os.path.join(current_dir, 'lane2d/evaluate'),
                         '-a', os.path.join(args.anno_dir, 'lane3d_1000/'),
                         '-d', args.pred_dir + '/',
                         '-i', os.path.join(args.anno_dir, 'images/'),
                         '-l', os.path.join(args.pred_dir, 'validation/test_list.txt'),
                         '-o', '/data/chenziye/HDMapNet/tools/metrics/openlane'])
    else:
        test_list = args.list[0]
        print(os.path.join(args.anno_dir, 'lane3d_1000/test', test_list))
        subprocess.call([os.path.join(current_dir, 'lane2d/evaluate'),
                         '-a', os.path.join(args.anno_dir, 'lane3d_1000/'),
                         '-d', args.pred_dir + '/',
                         '-i', os.path.join(args.anno_dir, 'images/'),
                         '-l', os.path.join(args.anno_dir, 'lane3d_1000/test', test_list),
                         '-o', './'])

    # for test_list in ['1000_curve.txt', '1000_intersection.txt', '1000_night.txt',
    #                   '1000_extreme_weather.txt', '1000_merge_split_case.txt', '1000_updown.txt']:
    #     print(os.path.join(args.anno_dir, 'lane3d_1000/test', test_list))
    #     subprocess.call([os.path.join(current_dir, 'lane2d/evaluate'),
    #                      '-a', os.path.join(args.anno_dir, 'lane3d_1000/'),
    #                      '-d', args.pred_dir + '/',
    #                      '-i', os.path.join(args.anno_dir, 'images/'),
    #                      '-l', os.path.join(args.anno_dir, 'lane3d_1000/test', test_list),
    #                      '-o', './'])


def parse_args():
    parser = argparse.ArgumentParser(description="Measure OpenLane's metric")
    parser.add_argument(
        "--pred-dir",
        default='/data/gpfs/projects/punim1962/project/HDMapNet/output/test_cond_lstr_2d_res34_openlane',
        help="Path to directory containing the predicted lanes"
    )
    parser.add_argument(
        "--anno-dir",
        default='/data/gpfs/projects/punim1962/datasets/openlane/',
        help="Path to directory containing the annotated lanes"
    )
    parser.add_argument("--top_view_region",
                        type=list,
                        default=[[-10, 103], [10, 103], [-10, 3], [10, 3]],
                        help="top_view_region")
    parser.add_argument("--list",
                        nargs='+',
                        default=[''],
                        help="Path to txt file containing the list of files"
    )
    parser.add_argument('--translate', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    main()
