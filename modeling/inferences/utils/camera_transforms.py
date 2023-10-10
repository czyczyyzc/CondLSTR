from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
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


def undistort_lane(lane, K, D, mode):
    new_lane = []
    src_pts = np.array(lane).reshape(-1, 1, 2)
    if mode != "fisheye":
        dst_pts = cv.undistortPoints(src=src_pts,
                                     cameraMatrix=K,
                                     distCoeffs=D,
                                     R=None,
                                     P=K)
        new_lane = dst_pts.reshape(-1).tolist()

    else:
        dst_pts = cv.fisheye.undistortPoints(
            distorted=src_pts,
            K=K,
            D=D,
            P=K,
            R=None)
        new_lane = dst_pts.reshape(-1).tolist()

    return new_lane


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
    lidar2ego_param_path = os.path.join(root_path, 'lidar_params/lidar_ego.json')
    with open(lidar2ego_param_path, "r") as f:
        lidar2ego = json.load(f)
    trans_lidar2ego = transform_matrix(np.array(list(lidar2ego['transform']['translation'].values())),
                                       Quaternion(list(lidar2ego['transform']['rotation'].values())), inverse=False)
    R_l2e = trans_lidar2ego[:3, :3]
    T_l2e = trans_lidar2ego[:3, 3]
    return R_l2e, T_l2e


def get_ego2cam_params(root_path, cam=15):
    cam_intrinsic_param_path = os.path.join(root_path, 'camera_params', 'camera_{}_intrinsic.json'.format(cam))
    lidar2cam_extrinsic_param_path = os.path.join(root_path, "camera_params", "camera_{}_extrinsic.json".format(cam))
    lidar2ego_param_path = os.path.join(root_path, "lidar_params", "lidar_ego.json")
    with open(cam_intrinsic_param_path, "r") as f:
        cam_intrinsic_params = json.load(f)
    with open(lidar2ego_param_path, "r") as f:
        lidar2ego = json.load(f)
    with open(lidar2cam_extrinsic_param_path, "r") as f:
        lidar2cam = json.load(f)
    K_cam2img = np.array(cam_intrinsic_params['K']).reshape(3, 3)
    D_img = np.array(cam_intrinsic_params['D'])
    img_distort_model = cam_intrinsic_params['distortion_model']
    trans_ego2lidar = transform_matrix(np.array(list(lidar2ego['transform']['translation'].values())),
                                       Quaternion(list(lidar2ego['transform']['rotation'].values())), inverse=True)
    trans_lidar2cam = transform_matrix(np.array(list(lidar2cam['transform']['translation'].values())),
                                       Quaternion(list(lidar2cam['transform']['rotation'].values())), inverse=False)
    trans_ego2cam = trans_lidar2cam @ trans_ego2lidar
    R_ego2cam = trans_ego2cam[:3, :3]
    T_ego2cam = trans_ego2cam[:3, 3]
    return R_ego2cam, T_ego2cam, K_cam2img, D_img, img_distort_model


def get_lidar2cam_params(root_path, cam=15):
    cam_intrinsic_param_path = os.path.join(root_path, 'camera_params', 'camera_{}_intrinsic.json'.format(cam))
    lidar2cam_extrinsic_param_path = os.path.join(root_path, "camera_params", "camera_{}_extrinsic.json".format(cam))
    with open(cam_intrinsic_param_path, "r") as f:
        cam_intrinsic_params = json.load(f)
    with open(lidar2cam_extrinsic_param_path, "r") as f:
        lidar2cam = json.load(f)
    trans_lidar2cam = transform_matrix(np.array(list(lidar2cam['transform']['translation'].values())),
                                       Quaternion(list(lidar2cam['transform']['rotation'].values())), inverse=False)
    R_lidar2cam = trans_lidar2cam[:3, :3]
    T_lidar2cam = trans_lidar2cam[:3, 3]
    return R_lidar2cam, T_lidar2cam


def get_calibration_sensor_data(root_path, cam_id=15):
    sensor_dict = {
        "camera_{}".format(cam_id): {},
        "lidar2ego": {}
    }
    cam_intrinsic_param_path = os.path.join(root_path, 'camera_params', 'camera_{}_intrinsic.json'.format(cam_id))
    lidar2cam_extrinsic_param_path = os.path.join(root_path, "camera_params", "camera_{}_extrinsic.json".format(cam_id))
    lidar2ego_param_path = os.path.join(root_path, "lidar_params", "lidar_ego.json")
    with open(cam_intrinsic_param_path, "r") as f:
        cam_intrinsic_params = json.load(f)
    with open(lidar2cam_extrinsic_param_path, "r") as f:
        lidar2cam = json.load(f)
    with open(lidar2ego_param_path, "r") as f:
        lidar2ego = json.load(f)
    sensor_dict["camera_{}".format(cam_id)]["extrinsic"] = lidar2cam["transform"]
    sensor_dict["camera_{}".format(cam_id)]["intrinsic"] = cam_intrinsic_params
    sensor_dict["lidar2ego"] = lidar2ego["transform"]
    return sensor_dict


# def project_ego2rv(p3d_ego, R_ego2cam, K_cam2img, T_ego2cam):
#     R_ego2img = K_cam2img @ R_ego2cam
#     T_ego2img = K_cam2img @ T_ego2cam
#     rv_points = (R_ego2img @ p3d_ego.T).T + T_ego2img.T
#     return rv_points


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


def project_rv2ego(rv_3d, R_ego2cam, K_cam2img, T_ego2cam):
    R_ego2img = K_cam2img @ R_ego2cam
    T_ego2img = K_cam2img @ T_ego2cam
    ego_points = np.linalg.inv(R_ego2img) @ (rv_3d - T_ego2img).T
    return ego_points.T


def project_ego2rv_v2(points_3d, calibration):
    trans_ego2img = calibration['trans_ego2img']
    points_3d = np.concatenate([points_3d, np.ones_like(points_3d[:, 0:1])], axis=1)
    points_2d = points_3d @ trans_ego2img.T
    points_2d = points_2d[:, :3]
    points_2d[:, :2] = points_2d[:, :2] / points_2d[:, 2:]
    return points_2d
