from pyquaternion import Quaternion
from refile import smart_open
import numpy as np
import cv2 as cv
import json
import os


def undistort_img(img, K, D, mode):
    h, w = img.shape[:2]
    if mode == 'pinhole':
        mapx, mapy = cv.initUndistortRectifyMap(K, D, None, K, (w, h), 5)
    elif mode == 'fisheye':
        mapx, mapy = cv.fisheye.initUndistortRectifyMap(K, D, None, K, (w, h), 5)
    return cv.remap(img, mapx, mapy, cv.INTER_LINEAR)


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


def get_lidar2ego_params(root_path):
    lidar2ego_param_path = os.path.join(root_path,'lidar_params/lidar_ego.json')
    with smart_open(lidar2ego_param_path,"r") as f:
        lidar2ego = json.load(f)
    trans_lidar2ego = transform_matrix(np.array(list(lidar2ego['transform']['translation'].values())),  Quaternion(list(lidar2ego['transform']['rotation'].values())), inverse=False)
    R_l2e = trans_lidar2ego[:3,:3]
    T_l2e = trans_lidar2ego[:3, 3]
    return R_l2e, T_l2e


def get_ego2cam_params(root_path, cam=15):
    cam_intrinsic_param_path = os.path.join(root_path,'camera_params','camera_{}_intrinsic.json'.format(cam))
    lidar2cam_extrinsic_param_path = os.path.join(root_path,"camera_params","camera_{}_extrinsic.json".format(cam))
    lidar2ego_param_path = os.path.join(root_path,"lidar_params","lidar_ego.json")
    with smart_open(cam_intrinsic_param_path,"r") as f:
        cam_intrinsic_params = json.load(f)
    with smart_open(lidar2ego_param_path,"r") as f:
        lidar2ego = json.load(f)
    with smart_open(lidar2cam_extrinsic_param_path,"r") as f:
        lidar2cam = json.load(f)
    K = np.array(cam_intrinsic_params['K']).reshape(3,3)
    D = np.array(cam_intrinsic_params['D'])
    model = cam_intrinsic_params['distortion_model']
    trans_ego2lidar = transform_matrix(np.array(list(lidar2ego['transform']['translation'].values())),
                     Quaternion(list(lidar2ego['transform']['rotation'].values())), inverse=True)
    trans_lidar2cam = transform_matrix(np.array(list(lidar2cam['transform']['translation'].values())),
                     Quaternion(list(lidar2cam['transform']['rotation'].values())), inverse=False)
    trans_ego2cam = trans_lidar2cam@trans_ego2lidar
    R_ego2cam = trans_ego2cam[:3,:3]
    T_ego2cam = trans_ego2cam[:3,3]
    return R_ego2cam, T_ego2cam, K, D, model


def get_ego2cam_params_v2(calibration):
    cam_intrinsic_params = calibration['cam_intrinsic']
    lidar2cam = calibration['cam_extrinsic']
    lidar2ego = calibration['lidar2ego']

    K = np.array(cam_intrinsic_params['K']).reshape(3, 3)
    D = np.array(cam_intrinsic_params['D'])
    model = cam_intrinsic_params['distortion_model']
    trans_ego2lidar = transform_matrix(np.array(list(lidar2ego['translation'].values())),
                                       Quaternion(list(lidar2ego['rotation'].values())), inverse=True)
    trans_lidar2cam = transform_matrix(np.array(list(lidar2cam['translation'].values())),
                                       Quaternion(list(lidar2cam['rotation'].values())), inverse=False)
    trans_ego2cam = trans_lidar2cam @ trans_ego2lidar
    R_ego2cam = trans_ego2cam[:3, :3]
    T_ego2cam = trans_ego2cam[:3, 3]
    return R_ego2cam, T_ego2cam, K, D, model


def project_ego2rv(p3d_ego,_r_ego2cam, _k_mat, _T_ego2cam):
    eff_R = _k_mat@_r_ego2cam
    eff_T = _k_mat@_T_ego2cam
    rv_points = (eff_R@p3d_ego.T).T + eff_T.T
    return rv_points


def project_rv2ego(rv_3d,_r_ego2cam, _k_mat, _T_ego2cam):
    eff_R = _k_mat@_r_ego2cam
    eff_T = _k_mat@_T_ego2cam
    ego_points = np.linalg.inv(eff_R)@(rv_3d-eff_T).T
    return ego_points

