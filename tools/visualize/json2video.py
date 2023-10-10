import os
import cv2
import json
import shutil
import refile
import pickle
import jsonlines
import nori2 as nori
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from collections import defaultdict


fpath_list = [
    # 's3://zjl-post-vis/0520_reformat_BEV_valid_ms.pckl',
    # 's3://zjl-post-vis/0520_reformat_BEV_valid_loadrv_fixextpitrol.pkl',
    # 's3://zjl-post-vis/0520_reformat_BEV_valid_ss_pretrained_fixextpitrol.pkl',
    # 's3://zjl-post-vis/0520_reformat_BEV_valid_ss_pretrained.pkl',
    # 's3://zjl-post-vis/0520_reformat_BEV_valid_loadrv.pkl'，
    # 's3://zjl-post-vis/0520_reformat_BEV_valid_ss_pretrained_fixextpitrol_-1.5.pkl'
    # 's3://czy1yzc/output/lane_points_0520_reformat_BEV_valid_bevdepth.pkl'
    's3://czy1yzc/output/lane_points_0520_reformat_BEV_valid.pkl',
]
cam_id = '0_6'
json_path = 's3://perceptor-share/data/lane_data_car2/0520_reformat_BEV_valid.json'


fetcher = nori.Fetcher()


def get_img_from_nid(nid):
    nid = nid.replace('.jpg', '')
    ns = np.frombuffer(fetcher.get(nid), dtype=np.uint8)
    # img = ns.reshape((1080, 1920, 3))
    img = cv2.imdecode(ns, cv2.IMREAD_COLOR)
    return img


class Undistort(object):

    def __init__(
            self,
            img_height=2160,
            img_width=3840,
            camera_ids=['0_6'],
            cam_calibration_path='s3://czy1yzc/calibresult/car_00002/',
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.cam_calibration_path = cam_calibration_path
        self.camera_ids = camera_ids
        self.camera_params = {}
        # load camera params
        self.load_camera_params()

    def load_camera_params(self):
        """ Load camera calibration parameters only when initialization """
        for cam_id in self.camera_ids:
            self.camera_params[cam_id] = {}

            root_path = self.cam_calibration_path
            cam_intrinsic_param_path = os.path.join(root_path, "camera_params",
                                                    "camera_{}_intrinsic.json".format(cam_id))
            with refile.smart_open(cam_intrinsic_param_path, "r") as f:
                cam_intrinsic_params = json.load(f)

            # with smart_open(self.cam_calibration_path, "r") as f:
            #     cam_calibration_data = json.load(f)
            #     cam_intrinsic_params = \
            #         cam_calibration_data['calibrated_sensors']['camera_{}'.format(cam_id)]['intrinsic']

            mode = cam_intrinsic_params['distortion_model']
            K = np.array(cam_intrinsic_params['K'], dtype=np.float32).reshape(3, 3)
            D = np.array(cam_intrinsic_params['D'], dtype=np.float32)
            self.camera_params[cam_id]['K'] = K
            self.camera_params[cam_id]['D'] = D
            self.camera_params[cam_id]['mode'] = mode
            # store undistort mapx,mapy
            if mode == 'pinhole':
                mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, (self.img_width, self.img_height), 5)
                self.camera_params[cam_id]['mapx'] = mapx
                self.camera_params[cam_id]['mapy'] = mapy
            elif mode == 'fisheye':
                mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, None, K, (self.img_width, self.img_height), 5)
                self.camera_params[cam_id]['mapx'] = mapx
                self.camera_params[cam_id]['mapy'] = mapy

    def __call__(self, img, cam_id='0_6'):
        """
        Undistort img &lane points using stored camera params
        """
        mapx = self.camera_params[cam_id]['mapx']
        mapy = self.camera_params[cam_id]['mapy']
        undistort_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        return undistort_img


undistort_img = Undistort()


def transform_matrix(
        translation: np.ndarray = np.array([0, 0, 0]),
        rotation: Quaternion = Quaternion([1, 0, 0, 0]),
        inverse: bool = False,
) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)
    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm


def project_ego2rv(points_3d, calibration):
    lidar2ego = calibration['lidar2ego']
    trans_ego2lidar = transform_matrix(
        np.array(list(lidar2ego['translation'].values())),
        Quaternion(list(lidar2ego['rotation'].values())), inverse=True)

    lidar2cam = calibration['cam_extrinsic']
    trans_cam2lidar = transform_matrix(
        np.array(list(lidar2cam['translation'].values())),
        Quaternion(list(lidar2cam['rotation'].values())), inverse=True)

    cam_intrin = calibration['cam_intrinsic']
    cam_K = np.array(cam_intrin['K']).reshape(3, 3)

    points_3d = np.concatenate([points_3d, np.ones_like(points_3d[:, 0:1])], axis=1)
    points_2d = points_3d @ trans_ego2lidar.T @ np.linalg.inv(trans_cam2lidar).T[:, :3] @ cam_K.T
    points_2d[:, :2] = points_2d[:, :2] / points_2d[:, 2:]
    return points_2d


json_dict = {}
with refile.smart_open(json_path, 'r') as f:
    file_data = json.load(f)

    calibrated_sensors = file_data['calibrated_sensors']
    calibration = {
        'cam_intrinsic': calibrated_sensors['camera_{}'.format(cam_id)]['intrinsic'],
        'cam_extrinsic': calibrated_sensors['camera_{}'.format(cam_id)]['extrinsic'],
        'lidar2ego':     calibrated_sensors['lidar2ego'],
    }

    frames = file_data['frame_data']
    for i, frame in enumerate(frames):
        nori_id = frame['rv']['camera_{}'.format(cam_id)]['nori_id']

        bev_lanes = []
        for idx, lane in frame['bev']['lane'].items():
            bev_lanes.append(np.array(lane['points']).reshape(-1, 3))

        rv_lanes = []
        for idx, lane in frame['rv']['camera_{}'.format(cam_id)]['lane'].items():
            rv_lanes.append(np.array(lane['points']).reshape(-1, 3))

        json_dict[nori_id] = {'index': i, 'bev_lanes': bev_lanes, 'rv_lanes': rv_lanes}


root = './test_demo_800w_boxun_car2_v2_5'


for i, fpath in enumerate(fpath_list):
    filename = fpath.split('/')[-1].replace('.json', '')
    filepath = os.path.join(root, filename)
    os.makedirs(filepath, exist_ok=True)

    count = 0
    with refile.smart_open(fpath, 'rb') as f:
        file_data = pickle.load(refile.smart_open(fpath, 'rb'))

        for item_dict in tqdm(file_data):
            nid = item_dict['nid']
            bev_lanes = item_dict['bev_lanes']

            img1 = get_img_from_nid(nid)
            img1 = undistort_img(img1)

            img2 = np.zeros((1080, 540, 3), dtype=np.uint8)
            img3 = np.zeros((1080, 540, 3), dtype=np.uint8)

            for j, lane in enumerate(bev_lanes):
                if 'lane_score' in item_dict and item_dict['lane_score'][j] < 0.5:
                    continue

                dt_lane_bev = np.array(lane).reshape(-1, 3)
                dt_lane_rv = project_ego2rv(dt_lane_bev, calibration)
                for p_curr, p_next in zip(dt_lane_rv[:-1], dt_lane_rv[1:]):
                    pt1 = (int(round(p_curr[0])), int(round(p_curr[1])))
                    pt2 = (int(round(p_next[0])), int(round(p_next[1])))
                    img1 = cv2.line(img1, pt1, pt2, color=(0, 0, 255), thickness=4)

                dt_lane_bev = (dt_lane_bev - np.array([65, 15, -3])) * np.array([-1080 / 60, -540 / 30, 540 / 6])
                for p_curr, p_next in zip(dt_lane_bev[:-1], dt_lane_bev[1:]):
                    pt1 = (int(round(p_curr[1])), int(round(p_curr[0])))
                    pt2 = (int(round(p_next[1])), int(round(p_next[0])))
                    img2 = cv2.line(img2, pt1, pt2, color=(0, 0, 255), thickness=4)

                    pt1 = (int(round(p_curr[2])), int(round(p_curr[0])))
                    pt2 = (int(round(p_next[2])), int(round(p_next[0])))
                    img3 = cv2.line(img3, pt1, pt2, color=(0, 0, 255), thickness=4)

            for lane in json_dict[nid]['rv_lanes']:
                gt_lane_rv = np.array(lane).reshape(-1, 3)[:, :2]
                for p_curr, p_next in zip(gt_lane_rv[:-1], gt_lane_rv[1:]):
                    pt1 = (int(round(p_curr[0])), int(round(p_curr[1])))
                    pt2 = (int(round(p_next[0])), int(round(p_next[1])))
                    img1 = cv2.line(img1, pt1, pt2, color=(255, 0, 0), thickness=4)

            for lane in json_dict[nid]['bev_lanes']:
                gt_lane_bev = np.array(lane).reshape(-1, 3)
                gt_lane_bev = (gt_lane_bev - np.array([65, 15, -3])) * np.array([-1080 / 60, -540 / 30, 540 / 6])
                for p_curr, p_next in zip(gt_lane_bev[:-1], gt_lane_bev[1:]):
                    pt1 = (int(round(p_curr[1])), int(round(p_curr[0])))
                    pt2 = (int(round(p_next[1])), int(round(p_next[0])))
                    img2 = cv2.line(img2, pt1, pt2, color=(255, 0, 0), thickness=4)

                    pt1 = (int(round(p_curr[2])), int(round(p_curr[0])))
                    pt2 = (int(round(p_next[2])), int(round(p_next[0])))
                    img3 = cv2.line(img3, pt1, pt2, color=(255, 0, 0), thickness=4)

            img1 = cv2.resize(img1, (1920, 1080))
            imgx = np.concatenate([img1, img2, img3], axis=1)
            out_path = os.path.join(filepath, '{}.jpg'.format(json_dict[nid]['index']))
            cv2.imwrite(out_path, imgx)

            count = count + 1

    os.system("ffmpeg  -i {}/%d.jpg -framerate 10 -vcodec libx264  -pix_fmt yuv420p -s 1920x1080 {}".format(
        filepath, os.path.join(root, filename + '.mp4')))
