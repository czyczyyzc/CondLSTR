import os
import cv2
import math
import json
import copy
import pickle
import argparse
import numpy as np
import timeout_decorator
from tqdm import tqdm
from os import path as osp
from concurrent.futures import ProcessPoolExecutor


DATASETS_TRAIN = {
    'v1.0': 'data_splits/standard/train.json',
    'v1.1': 'data_splits/rare_subset/train.json',
    'v1.2': 'data_splits/illus_chg/train.json',
}

DATASETS_VAL = {
    'v1.0': 'data_splits/standard/test.json',
    'v1.1': 'data_splits/rare_subset/test.json',
    'v1.2': 'data_splits/illus_chg/test.json',
}


def lane_data_prep(root_path,
                   info_prefix,
                   version='v1.0',
                   num_workers=1):
    """Create info file of nuscene dataset.
    Given the raw data, generate its related info file in pkl format.
    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        num_workers: int
    """
    thickness = 16

    train_data_infos = []
    for key, fpath in DATASETS_TRAIN.items():
        file_path = os.path.join(root_path, fpath)
        with open(file_path, 'r') as file:
            train_data_samples = [line for line in file]
        print('train samples: {}'.format(len(train_data_samples)))
        for i in range(int(math.ceil(len(train_data_samples) / 3200))):
            train_samples = copy.deepcopy(train_data_samples[i * 3200: (i + 1) * 3200])
            train_infos = INfoProcessor(root_path, train_samples, thickness, num_workers)()
            print('train infos: {}'.format(len(train_infos)))
            train_data_infos.extend(train_infos)
    print('Total train infos: {}'.format(len(train_data_infos)))

    val_data_infos = []
    for key, fpath in DATASETS_VAL.items():
        file_path = os.path.join(root_path, fpath)
        with open(file_path, 'r') as file:
            val_data_samples = [line for line in file]
        print('val samples: {}'.format(len(val_data_samples)))
        for i in range(int(math.ceil(len(val_data_samples) / 3200))):
            val_samples = copy.deepcopy(val_data_samples[i * 3200: (i + 1) * 3200])
            val_infos = INfoProcessor(root_path, val_samples, thickness, num_workers)()
            print('val infos: {}'.format(len(val_infos)))
            val_data_infos.extend(val_infos)
    print('Total val infos: {}'.format(len(val_data_infos)))

    metadata = dict(version=version)
    data = dict(infos=train_data_infos, metadata=metadata)
    info_train_path = osp.join(root_path, '{}_infos_train_{}.pkl'.format(info_prefix, version))
    with open(info_train_path, 'wb') as file:
        pickle.dump(data, file)
    
    data = dict(infos=val_data_infos, metadata=metadata)
    info_val_path = osp.join(root_path, '{}_infos_val_{}.pkl'.format(info_prefix, version))
    with open(info_val_path, 'wb') as file:
        pickle.dump(data, file)
    return


class INfoProcessor(object):

    def __init__(self, root, samples, thickness=8, num_workers=1):
        self.root = root
        self.samples = samples
        self.thickness = thickness
        self.num_workers = num_workers

    def _process_single(self, worker_id):
        samples = self.samples[worker_id::self.num_workers]
        data_infos = []
        for i, sample in enumerate(tqdm(samples)):
            # try:
            img_info = self.obtain_img_info(sample)
            if isinstance(img_info, list):
                data_infos.extend(img_info)
            elif isinstance(img_info, dict):
                data_infos.append(img_info)
            # except:
            #     continue
        return data_infos

    # @timeout_decorator.timeout(10, timeout_exception=StopIteration)
    def obtain_img_info(self, sample):
        info_dict = json.loads(sample)

        img_path = os.path.join(self.root, info_dict['raw_file'])
        assert os.path.exists(img_path), '{:s} not exist'.format(img_path)

        img_name = '_'.join(img_path.split('/')[-2:])
        img = cv2.imread(img_path)

        gt_lanes = info_dict['laneLines']
        gt_visibility = info_dict['laneLines_visibility']
        for i, lane in enumerate(gt_lanes):
            # A GT lane can be either 2D or 3D
            # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
            lane = np.array(lane)
            gt_lanes[i] = lane
            gt_visibility[i] = np.array(gt_visibility[i])
        if 'category' in info_dict:
            gt_category = info_dict['category']
        else:
            gt_category = [1 for _ in range(len(gt_lanes))]

        cam_intrinsics = np.array([[2015., 0., 960.],
                                   [0., 2015., 540.],
                                   [0., 0., 1.]])

        cam_pitch = info_dict['cam_pitch']
        cam_height = info_dict['cam_height']
        P_g2c = np.array([[1, 0, 0, 0],
                          [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                          [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch), 0],
                          [0, 0, 0, 1]])
        cam_extrinsics = np.linalg.inv(P_g2c)

        # cam_K = cam_intrinsics
        # cam_E = cam_extrinsics
        # P_g2im = projection_g2im_extrinsic(cam_E, cam_K)
        # H_g2im = homograpthy_g2im_extrinsic(cam_E, cam_K)
        # H_im2g = np.linalg.inv(H_g2im)
        # P_g2gflat = np.matmul(H_im2g, P_g2im)

        # prune gt lanes by visibility labels
        # gt_lanes = [prune_3d_lane_by_visibility(gt_lane, gt_visibility[k]) for k, gt_lane in enumerate(gt_lanes)]

        # prune out-of-range points are necessary before transformation
        # gt_lanes = [prune_3d_lane_by_range(gt_lane, 3*self.x_min, 3*self.x_max) for gt_lane in gt_lanes]

        # prune gt lanes by visibility labels
        gt_lanes = [gt_lane[gt_visibility[k] > 0, ...] for k, gt_lane in enumerate(gt_lanes)]

        gt_category = [gt_category[i] for i, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        # # convert 3d lanes to flat ground space
        # convert_lanes_3d_to_gflat(gt_lanes, P_g2gflat)
        #
        # gt_lanes_ = []
        # gt_category_ = []
        # for i in range(len(gt_lanes)):
        #     gt_lane_3d = gt_lanes[i]
        #     # prune out points not in valid range, requires additional points to interpolate better
        #     # prune out-of-range points after transforming to flat ground space, update visibility vector
        #     valid_indices = np.logical_and(np.logical_and(gt_lane_3d[:, 1] > 0, gt_lane_3d[:, 1] < 200),
        #                                    np.logical_and(gt_lane_3d[:, 0] > 3 * self.x_min,
        #                                                   gt_lane_3d[:, 0] < 3 * self.x_max))
        #     gt_lane_3d = gt_lane_3d[valid_indices, ...]
        #     # use more restricted range to determine deletion or not
        #     if gt_lane_3d.shape[0] < 2 or np.sum(np.logical_and(gt_lane_3d[:, 0] > self.x_min,
        #                                                         gt_lane_3d[:, 0] < self.x_max)) < 2:
        #         continue
        #
        #     # only keep the portion y is monotonically increasing above a threshold, to prune those super close points
        #     gt_lane_3d = make_lane_y_mono_inc(gt_lane_3d)
        #     if gt_lane_3d.shape[0] < 2:
        #         continue
        #
        #     gt_lanes_.append(gt_lane_3d.tolist())
        #     gt_category_.append(gt_category[i])
        #
        # gt_lanes = gt_lanes_
        # gt_category = gt_category_

        calibration = {
            'intrinsics': cam_intrinsics.tolist(),
            'extrinsics': cam_extrinsics.tolist(),
        }

        img_info = {
            'img_name':    img_name,
            'img_path':    img_path,
            'img_shape':   img.shape,
            'lane_points': gt_lanes,
            'lane_attris': gt_category,
            'calibration': calibration,
        }
        # # Only for test
        # import shutil
        # path = '/data/sets/img_mask/'
        # shutil.rmtree(path, ignore_errors=True)
        # os.makedirs(path, exist_ok=True)
        # img[img_mask > 0] = np.array([0, 0, 255], dtype=np.uint8)
        # image = Image.fromarray(img[:, :, ::-1], mode="RGB")
        # image.save(osp.join(path, '{}.jpg'.format(img_name)))
        # import sys
        # sys.exit()
        return img_info

    def __call__(self):
        if self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                data_infos = list(executor.map(self._process_single, range(self.num_workers)))
            data_infos_all = []
            for data_infos_ in data_infos:
                data_infos_all.extend(data_infos_)
        else:
            data_infos_all = self._process_single(0)
        return data_infos_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preprocessing for nuScenes dataset.')
    parser.add_argument('--root', type=str, default='/data/sets/apollo_sim')
    parser.add_argument('--version', type=str, default='v1.0')
    parser.add_argument('--num-workers', type=int, default=32)

    args = parser.parse_args()

    lane_data_prep(
        root_path=args.root,
        info_prefix='apollo_sim',
        version=args.version,
        num_workers=args.num_workers
    )
