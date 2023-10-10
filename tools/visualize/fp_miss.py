import os
import cv2
import json
import copy
import refile
import random
import colorsys
import jsonlines
import numpy as np
import nori2 as nori
from refile import smart_open
from pyquaternion import Quaternion


class Visualizer(object):

    ego_correction = {
        10: {
            'roll': 0,
            'yaw': 0.5,
            'pitch': -1.6,
        },
        11: {
            'roll': 0,
            'yaw': 0.5,
            'pitch': -1.55,
        },
        15: {
            'roll': 0,
            'yaw': 0,
            'pitch': -1.5,
        },
        13: {
            'roll': 0,
            'yaw': 0,
            'pitch': -1.5,
        },
        14: {
            'roll': 0,
            'yaw': 0,
            'pitch': -1.5,
        },
        12: {
            'roll': 0,
            'yaw': 0,
            'pitch': -1.5,
        }
    }

    def __init__(self, eval_file, pred_file, anno_file, save_root, cam_calibration_path):
        self.eval_file = eval_file
        self.pred_file = pred_file
        self.anno_file = anno_file
        self.save_root = save_root
        self.cam_calibration_path = cam_calibration_path
        self.height = 1080
        self.width = 1920
        self.bev_height = 1024
        self.bev_width = 1024
        self.ego_points = [-20, 20, 3, 43]
        self.camera_ids = [10, 11, 12, 13, 14, 15]
        self.camera_params = {}
        self.load_camera_params()
        self.fetcher = nori.Fetcher()
        os.makedirs(save_root, exist_ok=True)

    def load_camera_params(self):
        """ Load camera calibration parameters only when initialization """
        for cam_id in self.camera_ids:
            self.camera_params[cam_id] = {}
            root_path = self.cam_calibration_path
            cam_intrinsic_param_path = os.path.join(root_path, "camera_params",
                                                    "camera_{}_intrinsic.json".format(cam_id))
            lidar2cam_extrinsic_param_path = os.path.join(root_path, "camera_params",
                                                          "camera_{}_extrinsic.json".format(cam_id))
            lidar2cam_intrinsic_param_path = os.path.join(root_path, "camera_params",
                                                          "camera_{}_intrinsic.json".format(cam_id))
            lidar2ego_param_path = os.path.join(root_path, "lidar_params", "lidar_ego.json")
            with smart_open(cam_intrinsic_param_path, "r") as f:
                cam_intrinsic_params = json.load(f)
            with smart_open(lidar2cam_extrinsic_param_path, "r") as f:
                lidar2cam_extrinsic_params = json.load(f)
            with smart_open(lidar2cam_intrinsic_param_path, "r") as f:
                lidar2cam_intrinsic_params = json.load(f)
            with smart_open(lidar2ego_param_path, "r") as f:
                lidar2ego_params = json.load(f)

            self.camera_params[cam_id]['R_cam2lidar'] = lidar2cam_extrinsic_params['transform']['rotation']
            self.camera_params[cam_id]['T_cam2lidar'] = lidar2cam_extrinsic_params['transform']['translation']
            self.camera_params[cam_id]['R_lidar2ego'] = lidar2ego_params['transform']['rotation']
            self.camera_params[cam_id]['T_lidar2ego'] = lidar2ego_params['transform']['translation']
            mode = cam_intrinsic_params['distortion_model']
            K = np.array(cam_intrinsic_params['K'], dtype=np.float32).reshape(3, 3)
            D = np.array(cam_intrinsic_params['D'], dtype=np.float32)
            self.camera_params[cam_id]['K'] = K
            self.camera_params[cam_id]['D'] = D
            self.camera_params[cam_id]['mode'] = mode
            # store undistort mapx,mapy
            if mode == 'pinhole':
                mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, (self.width, self.height), 5)
                self.camera_params[cam_id]['mapx'] = mapx
                self.camera_params[cam_id]['mapy'] = mapy
            elif mode == 'fisheye':
                mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, None, K, (self.width, self.height), 5)
                self.camera_params[cam_id]['mapx'] = mapx
                self.camera_params[cam_id]['mapy'] = mapy

            # load ego self-correction params
            # only cam-10, cam-11, cam-15 are fine-corrected, other-cams need further correction to fuse global-bev
            self.camera_params[cam_id]['ego_roll_correction'] = self.ego_correction[cam_id]['roll']
            self.camera_params[cam_id]['ego_pitch_correction'] = self.ego_correction[cam_id]['pitch']
            self.camera_params[cam_id]['ego_yaw_correction'] = self.ego_correction[cam_id]['yaw']

    @staticmethod
    def obtain_colors(N, shuffle=False, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        if shuffle:
            random.shuffle(colors)
        colors = (np.asarray(colors) * 255).astype(np.uint8)
        return colors

    def get_img_from_nid(self, nid):
        img = np.frombuffer(self.fetcher.get(nid), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return img

    def undistort_img(self, img, cam_id):
        """
        Undistort img &lane points using stored camera params
        """
        mapx = self.camera_params[cam_id]['mapx']
        mapy = self.camera_params[cam_id]['mapy']
        undistort_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        return undistort_img

    def get_e2e_RT_matrices(self, theta_x, theta_y, theta_z, tx, ty, tz, cam_id):
        """
        get ego --> ego' rotation, translation matrix
        theta_x: pitch
        theta_y: roll
        theta_z: ego-car orientation rotation, yaw
        """
        R = np.zeros((3, 3))
        T_vec = np.zeros(3)
        """ other cameras are center-front/back cams, rotate clockwise or counter-clockwise """
        if cam_id not in [10, 11, 2, 5]:
            if np.random.rand() < 0.5:
                theta_z = (np.random.rand() - 0.5) * theta_z / 180.0 * np.pi
            else:
                theta_z = 0
        """ cam-10 & cam-2 are front-right/back-left cams, rotate counter-clockwise """
        if cam_id == 10 or cam_id == 2:
            if np.random.rand() < 0.5:
                theta_z = -0.5 * np.random.rand() * theta_z / 180.0 * np.pi
            else:
                theta_z = 0
        """ cam-11 & cam-5 are front-left/back-right cams, rotate clockwise """
        if cam_id == 11 or cam_id == 5:
            if np.random.rand() < 0.5:
                theta_z = 0.5 * np.random.rand() * theta_z / 180.0 * np.pi
            else:
                theta_z = 0

        if np.random.rand() < 0.5:
            theta_x = (np.random.rand() - 0.5) * theta_x / 180.0 * np.pi
        else:
            theta_x = 0
        if np.random.rand() < 0.5:
            theta_y = (np.random.rand() - 0.5) * theta_y / 180.0 * np.pi
        else:
            theta_y = 0

        """ camera-correction """
        theta_x += self.camera_params[cam_id]['ego_pitch_correction'] * np.pi / 180.0
        theta_y += self.camera_params[cam_id]['ego_roll_correction'] * np.pi / 180.0
        theta_z += self.camera_params[cam_id]['ego_yaw_correction'] * np.pi / 180.0

        T_vec[0] = (np.random.rand() - 0.5) * tx
        T_vec[1] = (np.random.rand() - 0.5) * ty
        T_vec[2] = (np.random.rand() - 0.5) * tz

        R_z = np.zeros((3, 3))
        R_z[0, 0] = np.cos(theta_z)
        R_z[1, 1] = np.cos(theta_z)
        R_z[2, 2] = 1
        R_z[0, 1] = np.sin(theta_z)
        R_z[1, 0] = -np.sin(theta_z)

        R_x = np.zeros((3, 3))
        R_x[0, 0] = np.cos(theta_x)
        R_x[2, 2] = np.cos(theta_x)
        R_x[0, 2] = np.sin(theta_x)
        R_x[2, 0] = -np.sin(theta_x)
        R_x[1, 1] = 1

        R_y = np.zeros((3, 3))
        R_y[0, 0] = 1
        R_y[1, 1] = np.cos(theta_y)
        R_y[2, 2] = np.cos(theta_y)
        R_y[1, 2] = np.sin(theta_y)
        R_y[2, 1] = -np.sin(theta_y)

        R = np.dot(R_z, np.dot(R_x, R_y))
        return R, T_vec

    def get_params(self, r_e2e, T_vec_e2e, cam_id):
        """
        get params for calculating warp matrix
        r_e2e_R: ego --> ego' rotation augmentation matrix
        T_vec_e2e: ego --> ego' translation augmentation matrix
        """
        R_cam2lidar = self.camera_params[cam_id]['R_cam2lidar']
        T_cam2lidar = self.camera_params[cam_id]['T_cam2lidar']
        R_lidar2ego = self.camera_params[cam_id]['R_lidar2ego']
        T_lidar2ego = self.camera_params[cam_id]['T_lidar2ego']
        K = self.camera_params[cam_id]['K']
        k_mat = np.array(K).reshape(3, 3)

        # camera-> Lidar rotation
        R_quat_c2l = [R_cam2lidar[key] for key in ['w', 'x', 'y', 'z']]
        r_c2l = Quaternion(R_quat_c2l).rotation_matrix

        # lidar -> ego rotation
        R_quat_l2e = [R_lidar2ego[key] for key in ['w', 'x', 'y', 'z']]
        r_l2e = Quaternion(R_quat_l2e).rotation_matrix

        T_vec_c2l = np.array([T_cam2lidar[key] for key in ['x', 'y', 'z']])
        T_vec_l2e = np.array([T_lidar2ego[key] for key in ['x', 'y', 'z']])

        # params for warp matrix calculation
        r_mat = np.dot(r_c2l, np.dot(r_l2e.T, r_e2e))
        T_vec = -np.dot(r_mat, (T_vec_l2e + np.dot(r_l2e.T, T_vec_e2e))) + T_vec_c2l
        return r_mat, k_mat, T_vec

    def get_warp_matrix(self, x1, x2, y1, y2, warp_w, warp_h, _r_mat, _k_mat, _T_vec, Z_w=0):
        """
        Get  camera --> virtual_camera(ego') warp/inv_warp matrix
        _r_mat: effective rotation matrix
        _T_vec: effective translation matrix
        _k_mat: camera_instrinsic param matrix
        warp_h: virtual_camera image height
        warp_w: virtual_camera image width
        x1,..y2: warp_rectangle, (y1,-x1), (y1,-x2), (y2,-x1), (y2,-x2)
        Z_w: ego' height
        """
        # virtual-camera is 270-degree rotated with respect to ego'-coordinates
        pt1 = np.array([y1, -x1, Z_w])
        pt2 = np.array([y1, -x2, Z_w])
        pt3 = np.array([y2, -x1, Z_w])
        pt4 = np.array([y2, -x2, Z_w])
        virtual_cam_points = [[0, warp_h], [warp_w, warp_h], [0, 0], [warp_w, 0]]
        ego_points = [pt1, pt2, pt3, pt4]
        cam_points = []
        for pt in ego_points:
            uv1_vector = np.dot(_k_mat, np.dot(_r_mat, pt.T)) + np.dot(_k_mat, _T_vec)
            Z_c = uv1_vector[2] + 1e-9
            cam_points.append([uv1_vector[0] / Z_c, uv1_vector[1] / Z_c])
        warp_mat = cv2.getPerspectiveTransform(np.float32(cam_points), np.float32(virtual_cam_points))
        inv_warp_mat = cv2.getPerspectiveTransform(np.float32(virtual_cam_points), np.float32(cam_points))
        return warp_mat, inv_warp_mat

    def __call__(self, *args, **kwargs):
        with refile.smart_open(self.anno_file, "r") as f:
            reader = jsonlines.Reader(f)
            samples = list(reader)

        with refile.smart_open(self.pred_file, "r") as f:
            reader = jsonlines.Reader(f)
            predicts = list(reader)[0]['pred']

        results = np.load(self.eval_file, allow_pickle=True)

        cam_id = 15
        for i, (sample, predict, result) in enumerate(zip(samples, predicts, results)):
            nori_id = sample['nori_id']
            img = self.get_img_from_nid(nori_id)
            undistort_img = self.undistort_img(img, cam_id)

            r_e2e_0, T_vec_e2e_0 = self.get_e2e_RT_matrices(0, 0, 0, 0, 0, 0, cam_id)
            # calculate warp matrices
            x1, x2, y1, y2 = self.ego_points
            # calculate warp matrix without rotation/translation
            r_mat_0, k_mat_0, T_vec_0 = self.get_params(r_e2e_0, T_vec_e2e_0, cam_id)
            warp_mat_0, inv_warp_mat_0 = self.get_warp_matrix(x1, x2, y1, y2, self.bev_width, self.bev_height, r_mat_0,
                                                              k_mat_0, T_vec_0, Z_w=-0.33)
            # warp perspective transform for imgs & lanes
            bev_aug_img = cv2.warpPerspective(undistort_img, warp_mat_0, (self.bev_width, self.bev_height))
            bev_aug_img = np.array(bev_aug_img)

            pr_list = predict[nori_id]
            gt_list = sample['per_pixel_label']

            miss_img = copy.deepcopy(bev_aug_img)
            for gt_points, hit in zip(gt_list, result['gt_list']):
                gt_points = np.array(gt_points).reshape(-1, 2)
                gt_points = (gt_points - np.array([-20, 43])) * np.array([1024/40, -1024/40])
                gt_points = gt_points.round().astype(np.int64)
                if hit:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                for p_curr, p_next in zip(gt_points[:-1], gt_points[1:]):
                    miss_img = cv2.line(miss_img, tuple(p_curr), tuple(p_next), color=color, thickness=10)
            cv2.imwrite(os.path.join(self.save_root, nori_id + '_miss' + '.jpg'), miss_img)

            fp_img = copy.deepcopy(bev_aug_img)
            for pr_points, hit in zip(pr_list, result['pr_list']):
                pr_points = np.array(pr_points).reshape(-1, 2)
                pr_points = (pr_points - np.array([-20, 43])) * np.array([1024/40, -1024/40])
                pr_points = pr_points.round().astype(np.int64)
                if hit:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                for p_curr, p_next in zip(pr_points[:-1], pr_points[1:]):
                    fp_img = cv2.line(fp_img, tuple(p_curr), tuple(p_next), color=color, thickness=10)
            cv2.imwrite(os.path.join(self.save_root, nori_id + '_fp' + '.jpg'), fp_img)


if __name__ == "__main__":
    # eval_file = "./feat_new.npy"
    # pred_file = "s3://czy1yzc/bev_cond_lstr_single/output/lane_points_new.json"
    # anno_file = "s3://colin/share/cam_15_test.json"

    eval_file = "./feat_perception_all.npy"
    pred_file = "s3://czy1yzc/bev_cond_lstr_single/output/lane_points_new_all.json"
    anno_file = "s3://colin/data/laneDet/changshu_2_cam_15_ego.json"

    save_root = "./temp"
    cam_calibration_path = "s3://czy1yzc/annotation/"
    visualizer = Visualizer(eval_file, pred_file, anno_file, save_root, cam_calibration_path)
    visualizer()
