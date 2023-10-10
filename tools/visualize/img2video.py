import os
import cv2
import json
import shutil
import refile
import pickle
import jsonlines
import numpy as np
from collections import defaultdict

fpath_list = [
    # 's3://tf-lane-data/videos/20220520/jsons/hard_U_turn_ppl_bag_20220208_111533_3dbmk__cq__02_0000_0001.json',
    # 's3://tf-lane-data/videos/20220520/jsons/hard_ramp_etc_ppl_bag_20220208_150348_3dbmk__gs__00_0000_0008.json',
    # 's3://tf-lane-data/videos/20220520/jsons/hard_roundabout_ppl_bag_20220208_111533_3dbmk__cqlk__00_0000_0007.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_T_junction_ppl_bag_20220209_071929_3dbmk__gs__01_0002_0004.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_bridge_ppl_bag_20220209_063602_3dbmk__gs__00_0000_0002.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_complex_ppl_bag_20220209_085806_3dbmk__cq__04_0004_0007.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_complex_ppl_bag_20220209_094201_3dbmk__cq__04_0008_0008.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_complex_ppl_bag_20220209_094201_3dbmk__cq__04_0011_0011.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_complex_ppl_bag_20220209_094201_3dbmk__cqlk__00_0000_0000.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_complex_ppl_bag_20220209_094201_3dbmk__cqlk__01_0000_0000.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_complex_ppl_bag_20220209_094201_3dbmk__cqlk__02_0000_0000.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_gradual_slope_ppl_bag_20220209_070434_3dbmk__gs__00_0000_0003.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_jam_ppl_bag_20220208_154634_3dbmk__gs__00_0014_0016.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_jam_ppl_bag_20220209_083609_3dbmk__cq__00_0000_0000.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_jam_ppl_bag_20220209_083609_3dbmk__cq__03_0000_0007.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_night_ppl_bag_20220208_180041_3dbmk__cq__04_0000_0000.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_night_ppl_bag_20220208_180041_3dbmk__cq__05_0000_0000.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_night_ppl_bag_20220208_180041_3dbmk__cq__06_0000_0000.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_night_ppl_bag_20220208_180041_3dbmk__cq__07_0000_0000.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_night_ppl_bag_20220208_180041_3dbmk__cq__09_0000_0000.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_night_ppl_bag_20220208_182019_3dbmk__cq__01_0018_0018.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_night_ppl_bag_20220208_192655_3dbmk__cqlk__04_0000_0002.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_ppl_bag_20220208_113033_3dbmk__cq__04_0000_0000.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_ppl_bag_20220208_150348_3dbmk__gs__01_0000_0010.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_ppl_bag_20220208_154634_3dbmk__gs__00_0004_0012.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_ppl_bag_20220209_063602_3dbmk__gs__00_0022_0025.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_ppl_bag_20220209_070434_3dbmk__gs__00_0008_0010.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_ppl_bag_20220209_083609_3dbmk__cq__01_0000_0004.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_ppl_bag_20220209_085806_3dbmk__cq__04_0000_0001.json',
    # 's3://tf-lane-data/videos/20220520/jsons/normal_ppl_bag_20220209_094201_3dbmk__cq__04_0014_0014.json',

    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_091756_3dbmk/02_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_091756_3dbmk/03_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_091756_3dbmk/04_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_091756_3dbmk/04_0001.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_091756_3dbmk/05_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_091756_3dbmk/06_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_091756_3dbmk/07_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_091756_3dbmk/07_0001.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_133434_3dbmk/03_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_133434_3dbmk/03_0001.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_133434_3dbmk/04_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_133434_3dbmk/05_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_133434_3dbmk/07_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_133434_3dbmk/08_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_133434_3dbmk/09_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_133434_3dbmk/10_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_133434_3dbmk/10_0001.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_133434_3dbmk/14_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_133434_3dbmk/15_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220526_133434_3dbmk/16_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220528_115314_3dbmk/01_0000.json",
    "s3://tf-labeled-res/20220616_BMK_checked/gszd/ppl_bag_20220530_110831_3dbmk/01_0001.json",
]
"""
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__00_0003_0013.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__01_0000_0003.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__04_0000_0004.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__08_0000_0001.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__100_0000_0001.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__105_0000_0002.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__106_0001_0005.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__10_0000_0002.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__122_0000_0001.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__129_0039_0042.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__129_0052_0054.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__20_0000_0004.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__52_0000_0001.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__78_0000_0000.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__90_0000_0000.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cq__91_0000_0000.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cqlk__14_0000_0001.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cqlk__44_0000_0000.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cqlk__67_0000_0000.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cqlk__82_0001_0001.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_050633_det__cqlk__98_0000_0000.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_124301_det__cq__27_0000_0002.json',
    's3://lbx-share/lane-vis/20220606/jsons/ppl_bag_20220520_124301_det__gs__00_0000_0002.json',
]
"""

root = '/data/chenziye/HDMapNet/output/test_cond_lstr_attr_result_car2_2'
root_ord = '/data/chenziye/HDMapNet/output/test_cond_lstr_attr_result_car2_2_ord'

for i, fpath in enumerate(fpath_list):
    # os.makedirs('test_pdemo_boxun_{}'.format(i), exist_ok=True)
    # os.makedirs('test_demo21_smooth1_ord', exist_ok=True)

    filename = '_'.join(fpath.split('/')[-2:]).replace('.json', '')
    filepath = os.path.join(root_ord, filename)
    os.makedirs(filepath, exist_ok=True)

    # result_pkl = []

    count = 0
    with refile.smart_open(fpath, 'r') as f:
        file_data = json.load(f)
        frames = file_data['frames']

        # frames = file_data['frame_data']
        for j, frame in enumerate(frames):
            nori_id = frame['sensor_data']['camera_0_6']['nori_id']
            # nori_id = frame['sensor_data']['camera_15']['nori_id']
            # nori_id = frame["rv"]["camera_15"]["nori_id"]
            # nori_id = nori_id.replace(',', '_')
            """
            reader = jsonlines.Reader(f)
            samples = list(reader)
            print(len(samples))
            for sample in samples:
            """

            try:
                img_path = os.path.join(root, '{}.jpg'.format(nori_id))
                out_path = os.path.join(filepath, '{}.jpg'.format(count))
                shutil.copyfile(img_path, out_path)
                count += 1
            except:
                continue

    os.system("ffmpeg  -i {}/%d.jpg -framerate 10 -vcodec libx264  -pix_fmt yuv420p -s 1920x1080 {}".format(
        filepath, os.path.join(root_ord, filename + '.mp4')))
