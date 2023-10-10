import os
import cv2
import mmcv
import refile
import jsonlines
import numpy as np
import nori2 as nori
import timeout_decorator
from concurrent.futures import ProcessPoolExecutor

fetcher = nori.Fetcher()


class INfoProcessor(object):

    def __init__(self, root, samples, num_workers=1):
        self.root = root
        self.samples = samples
        self.num_workers = num_workers

    def _process_single(self, worker_id):
        samples = self.samples[worker_id::self.num_workers]
        data_infos = []
        for i, sample in enumerate(mmcv.track_iter_progress(samples)):
            try:
                img_info = self.obtain_img_info(sample)
                if isinstance(img_info, list):
                    data_infos.extend(img_info)
                elif isinstance(img_info, dict):
                    data_infos.append(img_info)
            except:  # StopIteration:
                continue
        return data_infos

    @timeout_decorator.timeout(5, timeout_exception=StopIteration)
    def obtain_img_info(self, sample):
        img_id = sample['nori_id']
        img = cv2.imdecode(np.frombuffer(fetcher.get(img_id), np.uint8), cv2.IMREAD_UNCHANGED)
        return sample

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


fpath_list = [
    # 's3://lumingjie-oss-hhb/data/2d_defused_data/changshu8.json',
    # 's3://lumingjie-oss-hhb/data/2d_defused_data/changshu9.json',
    # 's3://lumingjie-oss-hhb/data/2d_defused_data/changshu10.json',
    # 's3://lumingjie-oss-hhb/data/2d_defused_data/changshu11.json',
    # 's3://lumingjie-oss-hhb/data/2d_defused_data/changshu12.json',
    # 's3://lumingjie-oss-hhb/data/2d_defused_data/changshu13.json',
    # 's3://lumingjie-oss-hhb/data/2d_defused_data/changshu14.json',
    # 's3://lumingjie-oss-hhb/data/2d_defused_data/changshu_nnmap3.json'
    's3://perceptor-share/data/lane_data/changshu3/cam_15.json',
    's3://perceptor-share/data/lane_data/changshu3_2/cam_15.json',
    's3://perceptor-share/data/lane_data/changshu4/cam_15.json',
    's3://perceptor-share/data/lane_data/changshu5/cam_15.json',
    's3://perceptor-share/data/lane_data/changshu6/cam_15.json',
    's3://perceptor-share/data/lane_data/changshu7/cam_15.json',
]

root_path = '/data/sets/new_json_files2'
num_workers = 32

os.makedirs(root_path, exist_ok=True)

for fpath in fpath_list:
    with refile.smart_open(fpath, "r") as f:
        reader = jsonlines.Reader(f)
        samples = list(reader)
        data_infos = INfoProcessor(root_path, samples, num_workers)()

    file_name = '_'.join(fpath.split('/')[-2:])
    with jsonlines.open(os.path.join(root_path, file_name), 'w') as f:
        for info in data_infos:
            f.write(info)
