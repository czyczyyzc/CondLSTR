import os
import cv2
import pickle
import shutil
import numpy as np
import torch
import torch.nn.functional as F
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


def smooth(points):
    kH = 21
    pt_all = torch.from_numpy(points)                           # (P, 3)
    pt_all = pt_all.t()[None]                                   # (1, 3, P)
    pt_avg = F.avg_pool1d(pt_all, kernel_size=kH, stride=1, padding=(kH - 1) // 2, count_include_pad=False)
    points = pt_avg[0].t().numpy()                              # (P, 3)
    return points


save_path = '/data/gpfs/projects/punim1962/project/HDMapNet/temp/test/150897849050449400.jpg'
pred_path = '/data/gpfs/projects/punim1962/project/HDMapNet/temp/test/results.pkl'
file_path = '/data/gpfs/projects/punim1962/datasets/openlane/images/validation/segment-15488266120477489949_3162_920_3182_920_with_camera_labels/150897849050449400.jpg'


img_data = cv2.imread(file_path)

with open(pred_path, 'rb') as f:
    results_list = pickle.load(f)


for result in results_list:
    res_list = result['2d_res']
    for idx, lane in enumerate(res_list):
        lane = [(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
        lane = np.array(lane)
        lane = smooth(lane)
        lane = lane.round().astype(np.int32)
        cv2.polylines(img_data, [lane], False, COLORS[idx + 1], thickness=8)
    cv2.imwrite(save_path, img_data)


