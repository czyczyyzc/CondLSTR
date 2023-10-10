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
from tqdm import tqdm
from refile import smart_open


class Visualizer(object):

    def __init__(self, eval_file, save_root, cam_calibration_path):
        self.eval_file = eval_file
        self.save_root = save_root
        self.cam_calibration_path = cam_calibration_path
        self.num_classes = 5
        self.height = 1080
        self.width = 1920
        self.bev_height = 1024
        self.bev_width = 1024
        self.ego_points = [-20, 20, 3, 43]
        self.camera_ids = [10, 11, 12, 13, 14, 15]
        self.camera_params = {}
        self.load_camera_params()
        self.fetcher = nori.Fetcher()
        self.lane_attributes = [['', 'road_curb_'], ['', 'guardrail_'], ['single_', 'double_'],
                                ['white_', 'yellow_'], ['dotted', 'solid']]
        os.makedirs(save_root, exist_ok=True)

    def load_camera_params(self):
        """ Load camera calibration parameters only when initialization """
        for cam_id in self.camera_ids:
            self.camera_params[cam_id] = {}
            root_path = self.cam_calibration_path
            cam_intrinsic_param_path = os.path.join(root_path, "camera_params",
                                                    "camera_{}_intrinsic.json".format(cam_id))
            with smart_open(cam_intrinsic_param_path, "r") as f:
                cam_intrinsic_params = json.load(f)

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

    def __call__(self, *args, **kwargs):
        results = np.load(self.eval_file, allow_pickle=True)[0]

        cam_id = 15
        for i, result in tqdm(enumerate(results)):
            nori_id = result['full']['img_meta']['nori_id']
            img = self.get_img_from_nid(nori_id)
            img = self.undistort_img(img, cam_id)

            pr_points_list = result['full']['pred_point']
            gt_points_list = result['full']['gt_point']
            pr_attrs_list = result['full']['pred_attr']
            gt_attrs_list = result['full']['gt_attr']
            pr_list_attr = result['full']['pr_list_attr']
            gt_list_attr = result['full']['gt_list_attr']
            pr_list = result['full']['pr_list']
            gt_list = result['full']['gt_list']

            if all([all(pr_attrs) for pr_attrs in pr_attrs_list] + [all(gt_attrs) for gt_attrs in gt_attrs_list]):
                continue

            attr_hit = [True for _ in range(self.num_classes)]

            miss_img = copy.deepcopy(img)
            for gt_points, gt_attrs, hit_attr, hit_gt in zip(gt_points_list, gt_attrs_list, gt_list_attr, gt_list):
                gt_points = np.array(gt_points).reshape(-1, 2)
                gt_points = gt_points.round().astype(np.int64)
                if all(hit_attr) and hit_gt:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                for p_curr, p_next in zip(gt_points[:-1], gt_points[1:]):
                    miss_img = cv2.line(miss_img, tuple(p_curr), tuple(p_next), color=color, thickness=4)

                center = tuple(gt_points[len(gt_points) // 2].round().astype(np.int64))
                attrib = ''
                for j, attr in enumerate(gt_attrs):
                    if attr == 255:
                        continue
                    attrib += self.lane_attributes[j][attr]
                    # if attrib == 'road_curb' or attrib == 'guardrail':
                    #     break
                miss_img = cv2.putText(miss_img, attrib, center, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                for j, attr in enumerate(hit_attr):
                    if not attr:
                        attr_hit[j] = False

            fp_img = copy.deepcopy(img)
            for pr_points, pr_attrs, hit_attr, hit_pr in zip(pr_points_list, pr_attrs_list, pr_list_attr, pr_list):
                pr_points = np.array(pr_points).reshape(-1, 2)
                pr_points = pr_points.round().astype(np.int64)
                if all(hit_attr) and hit_pr:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                for p_curr, p_next in zip(pr_points[:-1], pr_points[1:]):
                    fp_img = cv2.line(fp_img, tuple(p_curr), tuple(p_next), color=color, thickness=4)

                center = tuple(pr_points[len(pr_points) // 2].round().astype(np.int64))
                attrib = ''
                for j, attr in enumerate(pr_attrs):
                    if attr == 255:
                        continue
                    attrib += self.lane_attributes[j][attr]
                    # if attrib == 'road_curb' or attrib == 'guardrail':
                    #     break
                fp_img = cv2.putText(fp_img, attrib, center, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                for j, attr in enumerate(hit_attr):
                    if not attr:
                        attr_hit[j] = False

            for j, attr in enumerate(attr_hit):
                os.makedirs(os.path.join(self.save_root, str(j)), exist_ok=True)
                if not attr:
                    cv2.imwrite(os.path.join(self.save_root, str(j), nori_id + '_miss' + '.jpg'), miss_img)
                    cv2.imwrite(os.path.join(self.save_root, str(j), nori_id + '_fp' + '.jpg'), fp_img)


if __name__ == "__main__":
    eval_file = "/data/temp_save_condlane.npy"
    save_root = "/data/temp"
    cam_calibration_path = "s3://czy1yzc/annotation/"
    visualizer = Visualizer(eval_file, save_root, cam_calibration_path)
    visualizer()
