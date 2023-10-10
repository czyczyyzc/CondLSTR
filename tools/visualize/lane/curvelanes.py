import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

shutil.rmtree('/data/chenziye/shows/', ignore_errors=True)
os.makedirs('/data/chenziye/shows/', exist_ok=True)

with open('/data/sets/curvelanes/valid/valid.txt', 'r') as f:
    lines = f.readlines()
    for i, line in tqdm(enumerate(lines)):
        if i in [0]:
            continue
        src_path0 = os.path.join('/data/chenziye/conditional-lane-detection-master/work_dirs/curvelanes/watch', 'images.' + line.strip().split('/')[-1] + '.gt.jpg')
        src_path1 = os.path.join('/data/chenziye/conditional-lane-detection-master/work_dirs/curvelanes/watch', 'images.' + line.strip().split('/')[-1])
        src_path2 = os.path.join('/data/chenziye/CLRNet/work_dirs/clr/r18_curvelanes/20221116_222705_lr_6e-04_b_24/shows/', line.strip().split('/')[-1])
        src_path3 = os.path.join('/data/chenziye/HDMapNet/output/test_cond_lstr_80lane_50epoch_old_great_curvelanes_s0.7/shows/', line.strip().split('/')[-1])
        dst_path1 = os.path.join('/data/chenziye/shows/', str(i) + '.jpg')

        src_data0 = cv2.imread(src_path0)
        src_data1 = cv2.imread(src_path1)
        src_data2 = cv2.imread(src_path2)
        src_data3 = cv2.imread(src_path3)

        src_data0 = cv2.resize(src_data0, (640, 360))
        src_data1 = cv2.resize(src_data1, (640, 360))
        src_data2 = cv2.resize(src_data2, (640, 360))
        src_data3 = cv2.resize(src_data3, (640, 360))

        dst_data1 = np.ones((360 * 2 + 10, 640 * 2 + 10, 3), dtype=np.uint8) * 255
        dst_data1[0 * 360: 1 * 360, 0 * 640: 1 * 640] = src_data0
        dst_data1[0 * 360: 1 * 360, 1 * 640 + 10: 2 * 640 + 10] = src_data1
        dst_data1[1 * 360 + 10: 2 * 360 + 10, 0 * 640: 1 * 640] = src_data2
        dst_data1[1 * 360 + 10: 2 * 360 + 10, 1 * 640 + 10: 2 * 640 + 10] = src_data3

        dst_data1 = cv2.putText(dst_data1, 'GT', (0 * 640, 0 * 360 + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        dst_data1 = cv2.putText(dst_data1, 'CondLane', (1 * 640 + 10, 0 * 360 + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        dst_data1 = cv2.putText(dst_data1, 'CLRNet', (0 * 640, 1 * 360 + 10 + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        dst_data1 = cv2.putText(dst_data1, 'Ours', (1 * 640 + 10, 1 * 360 + 10 + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imwrite(dst_path1, dst_data1)


