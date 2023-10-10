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


def projective_transformation(Matrix, x, y, z, uv=False):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
            uv (bool): whether transform to image plane or not
    """
    ones = np.ones((1, len(z)))
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    if uv:
        x_vals = trans[0, :] / trans[2, :]
        y_vals = trans[1, :] / trans[2, :]
        return x_vals, y_vals
    else:
        x_vals = trans[0, :]
        y_vals = trans[1, :]
        z_vals = trans[2, :]
        return x_vals, y_vals, z_vals


def homograpthy_g2im_extrinsic(E, K):
    """E: extrinsic matrix, 4*4"""
    E_inv = np.linalg.inv(E)[0:3, :]
    H_g2c = E_inv[:, [0, 1, 3]]
    H_g2im = np.matmul(K, H_g2c)
    return H_g2im


def projection_g2im_extrinsic(E, K):
    E_inv = np.linalg.inv(E)[0:3, :]
    P_g2im = np.matmul(K, E_inv)
    return P_g2im


def smooth(lane):
    # interpolate
    points = lane[lane[:, 1].argsort()]
    point_ = []
    for (p_curr, p_next) in zip(points[:-1], points[1:]):
        inter_num = int(np.sqrt(((p_curr - p_next) ** 2).sum()) / 0.5) + 1
        point_new = np.linspace(p_curr, p_next, num=inter_num, endpoint=False)
        point_.extend(list(point_new))
    point_.append(points[-1])
    lane = np.array(point_).reshape(-1, 3)

    # smooth
    if len(lane) < 1:
        return []
    points = torch.from_numpy(lane)                                      # (P, 3)
    point_ = points.t()[None]                                            # (1, 3, P)
    count_ = torch.ones_like(point_[:, :1])                              # (1, 1, P)
    point_ = F_.pad(point_, pad=(kH - 1, kH - 1), mode='constant')       # (1, 3, P)
    count_ = F_.pad(count_, pad=(kH - 1, kH - 1), mode='constant')       # (1, 1, P)
    point_ = F_.avg_pool1d(point_, kernel_size=kH, stride=1, padding=0)  # (1, 3, P)
    count_ = F_.avg_pool1d(count_, kernel_size=kH, stride=1, padding=0)  # (1, 1, P)
    point_ = point_ / count_                                             # (1, 3, P)
    lane = point_[0].t().numpy()                                         # (P, 3)
    return lane


def draw_img_mask(img_data, lane, P_ego2img, color=(0, 0, 255)):
    # draw image mask
    x_2d, y_2d = projective_transformation(
        P_ego2img, lane[:, 0], lane[:, 1], lane[:, 2], uv=True)
    points = np.stack([x_2d, y_2d], axis=1)
    points = points[(x_2d >= 0) & (x_2d < img_data.shape[1]) &
                    (y_2d >= 0) & (y_2d < img_data.shape[0])]
    for (p_curr, p_next) in zip(points[:-1], points[1:]):
        p_curr = (int(round(p_curr[0])), int(round(p_curr[1])))
        p_next = (int(round(p_next[0])), int(round(p_next[1])))
        img_data = cv2.line(img_data, p_curr, p_next, color=color, thickness=8)
    return img_data


def draw_bev_mask(bev_data, lane, P_ego2ipm, color=(0, 0, 255)):
    # draw bev mask
    x_2d, y_2d = projective_transformation(
        P_ego2ipm, lane[:, 0], lane[:, 1], lane[:, 2], uv=True)
    points = np.stack([x_2d, y_2d], axis=1)
    points = points[(x_2d >= 0) & (x_2d < bev_data.shape[1]) &
                    (y_2d >= 0) & (y_2d < bev_data.shape[0])]
    for (p_curr, p_next) in zip(points[:-1], points[1:]):
        p_curr = (int(round(p_curr[0])), int(round(p_curr[1])))
        p_next = (int(round(p_next[0])), int(round(p_next[1])))
        bev_data = cv2.line(bev_data, p_curr, p_next, color=color, thickness=8)
    return bev_data


data_path0 = '/data/gpfs/projects/punim1962/datasets/once_3dlanes/data'
file_path0 = '/data/gpfs/projects/punim1962/datasets/once_3dlanes/test'
file_path1 = '/data/gpfs/projects/punim1962/project/PersFormer_3DLane/data_splits/once/persformer_once/once_pred/test'
file_path2 = '/data/gpfs/projects/punim1962/project/HDMapNet/output/test_bev_cond_lstr_attr_once_proj_3d_swin_2/test'
save_path1 = '/data/gpfs/projects/punim1962/once_shows/persformer/'
save_path2 = '/data/gpfs/projects/punim1962/once_shows/ours/'
file_list0 = glob.glob(os.path.join(file_path0, '*/*/*.json'), recursive=True)
file_list1 = glob.glob(os.path.join(file_path1, '*/*/*.json'), recursive=True)
file_list2 = glob.glob(os.path.join(file_path2, '*/*/*.json'), recursive=True)

shutil.rmtree('/data/gpfs/projects/punim1962/once_shows', ignore_errors=True)
random.seed(100)
kH = 21
os.makedirs(save_path1, exist_ok=True)
os.makedirs(save_path2, exist_ok=True)

# random.shuffle(file_list0)
# file_list0 = file_list0[:200]

for idx in tqdm(range(len(file_list0))):
    ann_path0 = file_list0[idx]

    if '1618716888300' in ann_path0:
        print(ann_path0)
    else:
        continue

    ann_name0 = '/'.join(ann_path0.split('/')[-3:])
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
        if len(file_lines) != 0:
            info_dict = json.loads(file_lines[0])
        else:
            print('Empty label_file: {}'.format(ann_path0))
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

    cam_intrinsics = info_dict['calibration']
    cam_intrinsics = np.array(cam_intrinsics)
    cam_intrinsics = cam_intrinsics[:, :3]

    # P_cam2img = cam_intrinsics
    # P_cam2ego = cam_extrinsics
    # P_ego2cam = np.linalg.inv(P_cam2ego)                 # (4, 4)
    # P_ego2img = np.matmul(P_cam2img, P_ego2cam[0:3, :])  # (3, 4)

    P_ego2img = projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
    H_ego2img = homograpthy_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
    H_img2ego = np.linalg.inv(H_ego2img)
    P_g2gflat = np.matmul(H_img2ego, P_ego2img)

    ipm_h, ipm_w = 800, 512
    top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
    H_ipm2ego = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                        [ipm_w - 1, 0],
                                                        [0, ipm_h - 1],
                                                        [ipm_w - 1, ipm_h - 1]]),
                                            np.float32(top_view_region))
    H_ego2ipm = np.linalg.inv(H_ipm2ego)
    H_img2ipm = np.linalg.inv(np.matmul(H_ego2img, H_ipm2ego))
    P_ego2ipm = np.matmul(H_ego2ipm, P_g2gflat)
    x_min, x_max, y_min, y_max, z_min, z_max = -10, 10, 3, 103, -5, 5

    bev_data0 = cv2.warpPerspective(img_data0, H_img2ipm, (ipm_w, ipm_h), flags=cv2.INTER_LINEAR)
    bev_data1 = copy.deepcopy(bev_data0)
    bev_data2 = copy.deepcopy(bev_data0)

    gt_lanes_packed = info_dict['lanes']

    gt_lanes = []
    for i, gt_lane_packed in enumerate(gt_lanes_packed):
        # A GT lane can be either 2D or 3D
        # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
        lane = np.array(gt_lane_packed).T
        # lane[2,:]=lane[2,:]/100.0 #TODO: check the unit of z

        # Coordinate convertion for openlane_300 data
        lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
        # cam_representation = np.linalg.inv(
        #                         np.array([[0, 0, 1, 0],
        #                                     [-1, 0, 0, 0],
        #                                     [0, -1, 0, 0],
        #                                     [0, 0, 0, 1]], dtype=float))  # transformation from apollo camera to openlane camera
        # lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))
        lane = np.matmul(cam_extrinsics, lane)

        lane = lane[0:3, :].T
        lane = lane[lane[:, 1].argsort()]  # TODO:make y mono increase

        # lane = smooth(lane)

        img_data1 = draw_img_mask(img_data1, lane, P_ego2img, color=(255, 0, 0))
        img_data2 = draw_img_mask(img_data2, lane, P_ego2img, color=(255, 0, 0))
        bev_data1 = draw_bev_mask(bev_data1, lane, P_ego2ipm, color=(255, 0, 0))
        bev_data2 = draw_bev_mask(bev_data2, lane, P_ego2ipm, color=(255, 0, 0))

        # img_data1 = draw_img_mask(img_data1, lane, P_ego2img, color=COLORS[i])
        # img_data2 = draw_img_mask(img_data2, lane, P_ego2img, color=COLORS[i])
        # bev_data1 = draw_bev_mask(bev_data1, lane, P_ego2ipm, color=COLORS[i])
        # bev_data2 = draw_bev_mask(bev_data2, lane, P_ego2ipm, color=COLORS[i])
        gt_lanes.append(lane)

    # with open(res_path1, 'r') as f:
    #     info_dict = json.load(f)
    # pr_lanes_packed = info_dict['lanes']
    #
    # pr_lane1 = []
    # for i, pr_lane_packed in enumerate(pr_lanes_packed):
    #     lane = np.array(pr_lane_packed['points'])[:, :3].T
    #     lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
    #     lane = np.matmul(cam_extrinsics, lane)
    #     lane = lane[0:3, :].T
    #     lane = lane[(lane[:, 0] >= x_min) & (lane[:, 0] <= x_max) &
    #                 (lane[:, 1] >= y_min) & (lane[:, 1] <= y_max) &
    #                 (lane[:, 2] >= z_min) & (lane[:, 2] <= z_max)]
    #
    #     # lane = smooth(lane)
    #     img_data1 = draw_img_mask(img_data1, lane, P_ego2img, color=(0, 0, 255))
    #     bev_data1 = draw_bev_mask(bev_data1, lane, P_ego2ipm, color=(0, 0, 255))
    #     pr_lane1.append(lane)
    #
    # img_path1 = os.path.join(save_path1, ann_name0.replace('.json', '_img.jpg'))
    # os.makedirs(os.path.dirname(img_path1), exist_ok=True)
    # cv2.imwrite(img_path1, img_data1)
    # cv2.imwrite(img_path1.replace('_img.jpg', '_bev.jpg'), bev_data1)

    with open(res_path2, 'r') as f:
        info_dict = json.load(f)
    pr_lanes_packed = info_dict['lanes']

    pr_lane2 = []
    for i, pr_lane_packed in enumerate(pr_lanes_packed):
        lane = np.array(pr_lane_packed['points'])[:, :3].T
        lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
        lane = np.matmul(cam_extrinsics, lane)
        lane = lane[0:3, :].T

        # lane = smooth(lane)
        img_data2 = draw_img_mask(img_data2, lane, P_ego2img, color=(0, 0, 255))
        bev_data2 = draw_bev_mask(bev_data2, lane, P_ego2ipm, color=(0, 0, 255))
        pr_lane2.append(lane)

    img_path2 = os.path.join(save_path2, ann_name0.replace('.json', '_img.jpg'))
    os.makedirs(os.path.dirname(img_path2), exist_ok=True)
    cv2.imwrite(img_path2, img_data2)
    cv2.imwrite(img_path2.replace('_img.jpg', '_bev.jpg'), bev_data2)

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

    # 3d_data2
    fig = plt.figure()  # plt.figure(figsize=(10,6), dpi=100)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-10, 10)
    ax.set_ylim(3, 103)
    ax.set_zlim(-2, 2)

    for i, lane in enumerate(gt_lanes):
        ax.plot(lane[:, 0], lane[:, 1], lane[:, 2], color='blue', label='gt' if i == 0 else '')
    for i, lane in enumerate(pr_lane2):
        ax.plot(lane[:, 0], lane[:, 1], lane[:, 2], color='red', label='pred' if i == 0 else '')

    ax.legend()
    plt.savefig(img_path2.replace('_img.jpg', '_3d.jpg'))
    plt.close()

