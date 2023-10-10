import os
import cv2
import math
import json
import copy
import pickle
import argparse
from tqdm import tqdm
from os import path as osp
from concurrent.futures import ProcessPoolExecutor


DATASETS_TRAIN = {
    'train': 's3://czyczyyzc/datasets/curvelanes/train.txt',
}

DATASETS_VAL = {
    'val': 's3://czyczyyzc/datasets/curvelanes/valid.txt',
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
        train_data_samples = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                img_path = os.path.join('train', line.strip())
                train_data_samples.append(img_path)
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
        val_data_samples = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                img_path = os.path.join('valid', line.strip())
                val_data_samples.append(img_path)
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
        img_path = os.path.join(self.root, sample)
        img_name = os.path.basename(img_path)

        img = cv2.imread(img_path)

        anno_path = img_path.replace('images/', 'labels/').replace('.jpg', '.lines.json')
        with open(anno_path, 'r') as f:
            data = json.load(f)['Lines']

        lanes = [[(float(point['x']), float(point['y'])) for point in lane] for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes if len(lane) >= 2]  # remove lanes with less than 2 points
        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y

        # img_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        # for idx, points in enumerate(lanes):
        #     for p_curr, p_next in zip(points[:-1], points[1:]):
        #         pt1 = (int(round(p_curr[0])), int(round(p_curr[1])))
        #         pt2 = (int(round(p_next[0])), int(round(p_next[1])))
        #         img_mask = cv2.line(img_mask, pt1, pt2, color=(idx + 1,), thickness=self.thickness)
        #
        # img_mask_path = osp.join(self.root, 'img_mask')
        # os.makedirs(img_mask_path, exist_ok=True)
        img_mask_path = osp.join('img_mask', img_name.replace('.jpg', '.npz'))
        # np.savez_compressed(osp.join(self.root, img_mask_path), img_mask)

        img_info = {
            'img_name':    img_name,
            'img_path':    img_path,
            'img_mask':    img_mask_path,
            'img_shape':   img.shape,
            'lane_points': lanes,
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
    parser.add_argument('--root', type=str, default='/data/sets/curvelanes')
    parser.add_argument('--version', type=str, default='v1.0')
    parser.add_argument('--num-workers', type=int, default=32)

    args = parser.parse_args()

    lane_data_prep(
        root_path=args.root,
        info_prefix='curvelanes',
        version=args.version,
        num_workers=args.num_workers
    )
