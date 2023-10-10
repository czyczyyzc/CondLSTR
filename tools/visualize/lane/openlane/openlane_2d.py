import os
import cv2
import copy
import glob
import json
import torch
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F_
from tqdm import tqdm


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]


def draw_img_mask(img_data, points, color=(0, 0, 255)):
    # draw image mask
    for (p_curr, p_next) in zip(points[:-1], points[1:]):
        p_curr = (int(round(p_curr[0])), int(round(p_curr[1])))
        p_next = (int(round(p_next[0])), int(round(p_next[1])))
        img_data = cv2.line(img_data, p_curr, p_next, color=color, thickness=8)
    return img_data


data_path0 = '/data/sets/openlane/images/validation'
file_path1 = '/data/chenziye/HDMapNet/output/test_img_bev_cond_lstr_attr_swap_data_aug_no_smooth_res18/validation'
file_path2 = '/data/chenziye/HDMapNet/output/test_img_bev_cond_lstr_attr_swap_data_aug_no_smooth_res18/validation'
file_list1 = glob.glob(os.path.join(file_path1, '**/*.json'), recursive=True)
file_list2 = glob.glob(os.path.join(file_path2, '**/*.json'), recursive=True)

shutil.rmtree('/data/chenziye/shows/', ignore_errors=True)
random.seed(100)
kH = 21

# for test_case in ['curve_case', 'merge_split_case', 'intersection_case']: # ['up_down_case']:
    # file_path0 = '/data/sets/openlane/lane3d_1000/test/' + test_case  # '/data/sets/openlane/lane3d_1000/validation'
    # save_path1 = '/data/chenziye/shows/persformer/' + test_case
    # save_path2 = '/data/chenziye/shows/ours/' + test_case

for test_case in ['validation']:
    file_path0 = '/data/sets/openlane/lane3d_1000/' + test_case  # '/data/sets/openlane/lane3d_1000/validation'
    save_path1 = '/data/chenziye/shows/ours/' + test_case
    save_path2 = '/data/chenziye/shows/ours/' + test_case
    file_list0 = glob.glob(os.path.join(file_path0, '**/*.json'), recursive=True)
    os.makedirs(save_path1, exist_ok=True)
    os.makedirs(save_path2, exist_ok=True)

    # random.shuffle(file_list0)
    # file_list0 = file_list0[:200]

    for idx in tqdm(range(len(file_list0))):
        ann_path0 = file_list0[idx]

        ann_name0 = '/'.join(ann_path0.split('/')[-2:])
        img_name0 = ann_name0.replace('.json', '.jpg')
        img_path0 = os.path.join(data_path0, img_name0)
        res_path1 = os.path.join(file_path1, ann_name0)
        res_path2 = os.path.join(file_path2, ann_name0)

        assert os.path.exists(res_path1)
        assert os.path.exists(res_path2)
        assert os.path.exists(ann_path0)
        assert os.path.exists(img_path0)

        img_data0 = cv2.imread(img_path0)
        img_data1 = copy.deepcopy(img_data0)
        img_data2 = copy.deepcopy(img_data0)

        with open(ann_path0, 'r') as file:
            file_lines = [line for line in file]
            info_dict = json.loads(file_lines[0])

        gt_lanes_packed = info_dict['lane_lines']

        gt_lanes = []
        for i, gt_lane_packed in enumerate(gt_lanes_packed):
            # A GT lane can be either 2D or 3D
            # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
            lane = np.array(gt_lane_packed['uv']).T

            # img_data1 = draw_img_mask(img_data1, lane, P_ego2img, color=(255, 0, 0))
            # img_data2 = draw_img_mask(img_data2, lane, P_ego2img, color=(255, 0, 0))
            # bev_data1 = draw_bev_mask(bev_data1, lane, P_ego2ipm, color=(255, 0, 0))
            # bev_data2 = draw_bev_mask(bev_data2, lane, P_ego2ipm, color=(255, 0, 0))

            print(lane)

            img_data1 = draw_img_mask(img_data1, lane, color=COLORS[i])
            img_data2 = draw_img_mask(img_data2, lane, color=COLORS[i])
            gt_lanes.append(lane)

        # with open(res_path1, 'r') as f:
        #     info_dict = json.load(f)
        # pr_lanes_packed = info_dict['lane_lines']
        #
        # pr_lane1 = []
        # for i, pr_lane_packed in enumerate(pr_lanes_packed):
        #     lane = np.array(pr_lane_packed['xyz'])[:, :3]
        #     lane = smooth(lane)
        #     img_data1 = draw_img_mask(img_data1, lane, P_ego2img, color=(0, 0, 255))
        #     bev_data1 = draw_bev_mask(bev_data1, lane, P_ego2ipm, color=(0, 0, 255))
        #     pr_lane1.append(lane)
        #
        # with open(res_path2, 'r') as f:
        #     info_dict = json.load(f)
        # pr_lanes_packed = info_dict['lane_lines']
        #
        # pr_lane2 = []
        # for i, pr_lane_packed in enumerate(pr_lanes_packed):
        #     lane = np.array(pr_lane_packed['xyz'])[:, :3]
        #     lane = smooth(lane)
        #     img_data2 = draw_img_mask(img_data2, lane, P_ego2img, color=(0, 0, 255))
        #     bev_data2 = draw_bev_mask(bev_data2, lane, P_ego2ipm, color=(0, 0, 255))
        #     pr_lane2.append(lane)

        img_path1 = os.path.join(save_path1, ann_name0.replace('.json', '_img.jpg'))
        os.makedirs(os.path.dirname(img_path1), exist_ok=True)
        cv2.imwrite(img_path1, img_data1)

        # img_path2 = os.path.join(save_path2, ann_name0.replace('.json', '_img.jpg'))
        # os.makedirs(os.path.dirname(img_path2), exist_ok=True)
        # cv2.imwrite(img_path2, img_data2)

        # # 3d_data1
        # fig = plt.figure()  # plt.figure(figsize=(10,6), dpi=100)
        # ax = fig.add_subplot(projection='3d')
        # ax.set_xlim(-10, 10)
        # ax.set_ylim(3, 103)
        # ax.set_zlim(-2, 2)
        #
        # for i, lane in enumerate(gt_lanes):
        #     ax.plot(lane[:, 0], lane[:, 1], lane[:, 2], color='blue', label='gt' if i == 0 else '')
        # for i, lane in enumerate(pr_lane1):
        #     ax.plot(lane[:, 0], lane[:, 1], lane[:, 2], color='red', label='pred' if i == 0 else '')
        #
        # ax.legend()
        # plt.savefig(img_path1.replace('_img.jpg', '_3d.jpg'))
        # plt.close()
        #
        # # 3d_data2
        # fig = plt.figure()  # plt.figure(figsize=(10,6), dpi=100)
        # ax = fig.add_subplot(projection='3d')
        # ax.set_xlim(-10, 10)
        # ax.set_ylim(3, 103)
        # ax.set_zlim(-2, 2)
        #
        # for i, lane in enumerate(gt_lanes):
        #     ax.plot(lane[:, 0], lane[:, 1], lane[:, 2], color='blue', label='gt' if i == 0 else '')
        # for i, lane in enumerate(pr_lane2):
        #     ax.plot(lane[:, 0], lane[:, 1], lane[:, 2], color='red', label='pred' if i == 0 else '')
        #
        # ax.legend()
        # plt.savefig(img_path2.replace('_img.jpg', '_3d.jpg'))
        # plt.close()

