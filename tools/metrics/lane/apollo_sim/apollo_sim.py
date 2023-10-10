import os
import sys
import cv2
import json
import pickle
import argparse
import jsonlines
import subprocess
import numpy as np
from tqdm import tqdm
from eval_3D_lane import *


def main():
    args = parse_args()

    print('#############################################################')
    print(args.pred_dir)
    print(args.anno_dir)

    if args.translate:
        with open(os.path.join(args.pred_dir, 'results.pkl'), 'rb') as f:
            results_list = pickle.load(f)

        with jsonlines.open(os.path.join(args.pred_dir, 'results.json'), mode='w') as json_writer:
            for result in tqdm(results_list):
                img_path = os.path.join(args.anno_dir, result['image_name'].replace(',', '/'))

                if not os.path.exists(img_path):
                    print(img_path)
                    continue
                lanelines_pred = result['3d_res']
                lanelines_conf = result['score']

                file_path = '/'.join(img_path.split('/')[-3:])
                result_dict = {
                    "raw_file": file_path,
                    "laneLines": lanelines_pred,
                    "laneLines_prob": lanelines_conf,
                }
                json_writer.write(result_dict)

    vis = False
    parser = define_args()
    eval_args = parser.parse_args()

    # two method are compared: '3D_LaneNet' and 'Gen_LaneNet'
    method_name = 'Gen_LaneNet_ext'

    # location where the original dataset is saved. Image will be loaded in case of visualization
    eval_args.dataset_dir = args.anno_dir

    # load configuration for certain dataset
    sim3d_config(eval_args)

    # Initialize evaluator
    evaluator = LaneEval(eval_args)

    # Three different splits of datasets: 'standard', 'rare_subsit', 'illus_chg'
    for data_split in ['standard', 'rare_subset', 'illus_chg']:
        # auto-file in dependent paths
        gt_file = os.path.join(args.anno_dir, 'data_splits/' + data_split + '/test.json')
        pred_file = os.path.join(args.pred_dir, 'results.json')

        # evaluation at varying thresholds
        eval_stats_pr = evaluator.bench_one_submit_varying_probs(pred_file, gt_file)
        max_f_prob = eval_stats_pr['max_F_prob_th']

        # evaluate at the point with max F-measure. Additional eval of position error.
        # Option to visualize matching result
        eval_stats = evaluator.bench_one_submit(pred_file, gt_file, prob_th=max_f_prob, vis=vis)

        print("Metrics: AP, F-score, x error (close), x error (far), z error (close), z error (far)")
        print("Laneline:  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}".format(eval_stats_pr['laneline_AP'],
                                                                           eval_stats[0], eval_stats[3],
                                                                           eval_stats[4], eval_stats[5],
                                                                           eval_stats[6]))
        # print("Centerline:  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}".format(eval_stats_pr['centerline_AP'],
        #                                                                      eval_stats[7], eval_stats[10],
        #                                                                      eval_stats[11], eval_stats[12],
        #                                                                      eval_stats[13]))


def parse_args():
    parser = argparse.ArgumentParser(description="Measure OpenLane's metric")
    parser.add_argument(
        "--pred_dir",
        default='/data/chenziye/HDMapNet/output/test_bev_cond_lstr_attr_data_aug_apollo_latest',
        help="Path to directory containing the predicted lanes"
    )
    parser.add_argument(
        "--anno_dir",
        default='/data/sets/apollo_sim/',
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
