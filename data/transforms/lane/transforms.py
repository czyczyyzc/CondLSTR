import cv2
from numpy import random
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from ..transforms import *


class GenerateLaneLine2D(object):

    def __init__(self, transforms, img_h, img_w, cut_height=0, thickness=4):
        assert isinstance(transforms, list), "transforms must be a list!"
        img_transforms = []
        for aug in transforms:
            p = aug['p']
            if aug['name'] != 'OneOf':
                img_transforms.append(
                    iaa.Sometimes(p=p,
                                  then_list=getattr(
                                      iaa,
                                      aug['name'])(**aug['parameters'])))
            else:
                img_transforms.append(
                    iaa.Sometimes(
                        p=p,
                        then_list=iaa.OneOf([
                            getattr(iaa,
                                    aug_['name'])(**aug_['parameters'])
                            for aug_ in aug['transforms']
                        ])))
        self.transform = iaa.Sequential(img_transforms)
        self.img_w, self.img_h = img_w, img_h
        self.cut_height = cut_height
        self.thickness = thickness

    @staticmethod
    def lane_to_linestrings(lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))
        return lines

    @staticmethod
    def linestrings_to_lanes(lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)
        return lanes

    def __call__(self, results):
        img_org = results['img']
        if isinstance(self.cut_height, int):
            cut_height = self.cut_height
        elif isinstance(self.cut_height, dict):
            cut_height = self.cut_height[img_org.shape]
        else:
            raise NotImplementedError

        img_org = img_org[cut_height:, :, :]

        lane_points = []
        for points in results['lane_points']:
            points = np.array(points)
            points = points[points[:, 1].argsort()]
            lane = []
            for p in points:
                lane.append((p[0], p[1] - cut_height))
            lane_points.append(lane)
        line_strings_org = self.lane_to_linestrings(lane_points)
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)

        img, line_strings = self.transform(
            image=img_org.copy().astype(np.uint8),
            line_strings=line_strings_org,
        )
        # line_strings.clip_out_of_image_()
        lane_points = self.linestrings_to_lanes(line_strings)

        if len(lane_points) > 0:
            img_mask = np.zeros((len(lane_points), img.shape[0], img.shape[1]), dtype=np.uint8)
            for i, points in enumerate(lane_points):
                for p_curr, p_next in zip(points[:-1], points[1:]):
                    pt1 = (int(round(p_curr[0])), int(round(p_curr[1])))
                    pt2 = (int(round(p_next[0])), int(round(p_next[1])))
                    img_mask[i] = cv2.line(img_mask[i], pt1, pt2, color=(1,), thickness=self.thickness)
            img_mask = img_mask.transpose((1, 2, 0))
            results['img_mask'] = img_mask

        if 'lane_attris' in results:
            lane_attris = results['lane_attris']
            assert len(lane_points) == len(lane_attris), \
                'Number of lanes {} and number of attributes {} are different'.format(len(lane_points), len(lane_attris))
            results['lane_attris'] = np.array(lane_attris, dtype=np.int64)

        org_h, org_w = img_org.shape[:2]
        new_h, new_w = img.shape[:2]
        w_scale = new_w / org_w
        h_scale = new_h / org_h
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['ori_shape'] = img_org.shape
        results['image_offset'] = np.array([0, cut_height])
        results['scale_factor'] = scale_factor
        results['lane_points'] = lane_points
        return results


class GenerateLaneLine3D(object):

    def __init__(self, img_h, img_w, bev_h, bev_w, bev_range, img_flip_p=0.5, img_rotate=10,
                 cut_height=0, theta_x=0, theta_y=0, theta_z=0, tx=0, ty=0, tz=0):
        self.img_w, self.img_h = img_w, img_h
        self.bev_w, self.bev_h = bev_w, bev_h
        self.img_flip_p = img_flip_p
        self.img_rotate = img_rotate

        self.x_min, self.y_min, self.x_max, self.y_max = bev_range
        self.bev_range = bev_range

        self.bev_scale = [self.bev_w / (self.x_max - self.x_min),
                          self.bev_h / (self.y_max - self.y_min),
                          10.0]

        self.P_ego2bev = np.array([[self.bev_scale[0], 0, 0, -self.x_min * self.bev_scale[0]],
                                   [0, self.bev_scale[1], 0, -self.y_min * self.bev_scale[1]],
                                   [0, 0, self.bev_scale[2], 0],
                                   [0, 0, 0, 1]], dtype=np.float32)

        self.P_bev2ego = np.linalg.inv(self.P_ego2bev)

        self.cut_height = cut_height
        self.theta_x, self.theta_y, self.theta_z = theta_x, theta_y, theta_z
        self.tx, self.ty, self.tz = tx, ty, tz

    def get_random_bev_trans(self):
        """
        get random ego --> ego' rotation, translation matrix
        theta_x: pitch
        theta_y: roll
        theta_z: yaw
        """
        P_ego2ego = np.eye(4)

        theta_x = random.uniform(-self.theta_x, self.theta_x) / 180.0 * np.pi
        theta_y = random.uniform(-self.theta_y, self.theta_y) / 180.0 * np.pi
        theta_z = random.uniform(-self.theta_z, self.theta_z) / 180.0 * np.pi

        R_x = np.zeros((3, 3))
        R_x[1, 1] = np.cos(theta_x)
        R_x[2, 2] = np.cos(theta_x)
        R_x[1, 2] = np.sin(theta_x)
        R_x[2, 1] = -np.sin(theta_x)
        R_x[0, 0] = 1

        R_y = np.zeros((3, 3))
        R_y[0, 0] = np.cos(theta_y)
        R_y[2, 2] = np.cos(theta_y)
        R_y[0, 2] = np.sin(theta_y)
        R_y[2, 0] = -np.sin(theta_y)
        R_y[1, 1] = 1

        R_z = np.zeros((3, 3))
        R_z[0, 0] = np.cos(theta_z)
        R_z[1, 1] = np.cos(theta_z)
        R_z[0, 1] = np.sin(theta_z)
        R_z[1, 0] = -np.sin(theta_z)
        R_z[2, 2] = 1

        """ Rotation matrix in the form of Euler angles (roll, pitch, yaw) """
        R_ego2ego = np.dot(R_z, np.dot(R_x, R_y))

        T_ego2ego = np.zeros(3)
        T_ego2ego[0] = random.uniform(-self.tx, self.tx)
        T_ego2ego[1] = random.uniform(-self.ty, self.ty)
        T_ego2ego[2] = random.uniform(-self.tz, self.tz)

        P_ego2ego[:3, :3] = R_ego2ego
        P_ego2ego[:3, 3] = T_ego2ego
        return P_ego2ego

    def get_random_img_trans(self, img_shape, results):
        H_img2img = np.eye(3)

        # Resize Aug
        org_h, org_w = img_shape[:2]
        new_h, new_w = self.img_h, self.img_w

        w_scale = new_w / org_w
        h_scale = new_h / (org_h - self.cut_height)

        rescale_mat = np.eye(3)
        rescale_mat[0, 0] = w_scale
        rescale_mat[1, 1] = h_scale
        rescale_mat[1, 2] = -h_scale * self.cut_height
        H_img2img = rescale_mat @ H_img2img

        img_shape = (new_h, new_w, 3)
        pad_shape = (new_h, new_w, 3)
        ori_shape = (org_h, org_w, 3)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

        results['img_shape'] = img_shape
        results['pad_shape'] = pad_shape
        results['ori_shape'] = ori_shape
        results['scale_factor'] = scale_factor
        results['image_offset'] = np.array([0, self.cut_height])

        # Flip Aug
        if np.random.rand() < self.img_flip_p:
            flip_mat = np.eye(3)
            flip_mat[0, 0] = -1
            flip_mat[0, 2] = new_w
            H_img2img = flip_mat @ H_img2img
            results['flip'] = True
        else:
            results['flip'] = False

        # Rotate Aug
        center_x = new_w / 2
        center_y = new_h / 2
        rot_deg = random.uniform(-self.img_rotate, self.img_rotate)
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rot_deg, 1.0)
        rot_mat = np.vstack([rot_mat, [0, 0, 1]])
        H_img2img = rot_mat @ H_img2img
        results['rot_angle'] = rot_deg
        return H_img2img

    @staticmethod
    def compute_plane_equation_XY(T):
        # 世界坐标系中的三个点
        O_w = np.array([0, 0, 0, 1])  # 原点
        A_w = np.array([1, 0, 0, 1])  # X轴上的点
        B_w = np.array([0, 1, 0, 1])  # Y轴上的点

        # 将这些点转换到相机坐标系中
        O_c = T.dot(O_w)
        A_c = T.dot(A_w)
        B_c = T.dot(B_w)

        # 计算两个向量
        OA_c = A_c - O_c
        OB_c = B_c - O_c

        # 计算法向量
        n = np.cross(OA_c[:3], OB_c[:3])

        # 根据平面方程得到常数d
        d = -n.dot(O_c[:3])
        return n, d

    @staticmethod
    def homographic_transformation(Matrix, x, y):
        """
        Helper function to transform coordinates defined by transformation matrix

        Args:
                Matrix (multi dim - array): 3x3 homography matrix
                x (array): original x coordinates
                y (array): original y coordinates
        """
        ones = np.ones((1, len(y)))
        coordinates = np.vstack((x, y, ones))
        trans = np.matmul(Matrix, coordinates)

        x_vals = trans[0, :] / trans[2, :]
        y_vals = trans[1, :] / trans[2, :]
        return x_vals, y_vals

    @staticmethod
    def projective_transformation(Matrix, x, y, z, uv=False):
        """
        Helper function to transform coordinates defined by transformation matrix

        Args:
                Matrix (multi dim - array): 3x4 projection matrix
                x (array): original x coordinates
                y (array): original y coordinates
                z (array): original z coordinates
                uv (bool): whether transform to image plane or not
        """
        ones = np.ones((1, len(z)))
        coordinates = np.vstack((x, y, z, ones))
        trans = np.matmul(Matrix, coordinates)

        if uv:
            x_vals = trans[0, :] / trans[2, :]
            y_vals = trans[1, :] / trans[2, :]
            z_vals = trans[2, :]
            return x_vals, y_vals, z_vals
        else:
            x_vals = trans[0, :]
            y_vals = trans[1, :]
            z_vals = trans[2, :]
            return x_vals, y_vals, z_vals

    @staticmethod
    def lane_interpolate(points):
        if len(points) == 1:
            return points
        points = points[np.argsort(points[:, 1])]
        
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        y_min, y_max = int(min(y)), int(max(y))
        
        y_new = np.arange(y_min, y_max + 1, 1)
        x_new = np.interp(y_new, y, x)
        z_new = np.interp(y_new, y, z)
        points = np.stack([x_new, y_new, z_new], axis=-1)
        return points

    def __call__(self, results):
        img_org = results['img']
        cam_intrinsics = np.array(results['calibration']['intrinsics'])
        cam_extrinsics = np.array(results['calibration']['extrinsics'])
        lane_points = [np.array(lane) for lane in results['lane_points']]
        
        H_img2img = self.get_random_img_trans(img_org.shape, results)
        P_ego2ego = self.get_random_bev_trans()                                          # (4, 4)
        P_cam2img = cam_intrinsics                                                       # (3, 3)
        P_cam2ego = cam_extrinsics                                                       # (4, 4) camera_old to world_old
        P_cam2img = np.matmul(H_img2img, P_cam2img)                                      # (3, 3) camera_old to image_new
        P_cam2ego = np.matmul(P_ego2ego, P_cam2ego)                                      # (4, 4) camera_old to world_new
        P_ego2cam = np.linalg.inv(P_cam2ego)                                             # (4, 4)
        P_ego2img = np.matmul(P_cam2img, P_ego2cam[0:3, :])                              # (3, 4)
        
        normal_vector, D = self.compute_plane_equation_XY(P_ego2cam)
        coords_v = np.arange(self.img_h, dtype=np.float32)                               # (H,)
        coords_u = np.arange(self.img_w, dtype=np.float32)                               # (W,)
        coords2d = np.stack(np.meshgrid(coords_u, coords_v, indexing='xy'), axis=0)      # (2, H, W)
        coords3d = np.concatenate((coords2d, np.ones_like(coords2d[:1])), axis=0)        # (3, H, W)
        coords3d = np.linalg.inv(P_cam2img) @ coords3d.reshape(3, -1)                    # (3, H * W)
        coords3d = (-D / (normal_vector[None] @ coords3d)) * coords3d                    # (3, H * W)
        coords3d = coords3d.reshape(3, self.img_h, self.img_w)                           # (3, H, W)
        depth_prior = coords3d[2]                                                        # (H, W)
        depth_prior = depth_prior.astype(np.float32)                                     # (H, W)
        
        gt_lanes = []
        for i, gt_lane in enumerate(lane_points):
            # change to new ego
            x_ego, y_ego, z_ego = self.projective_transformation(
                P_ego2ego, gt_lane[:, 0], gt_lane[:, 1], gt_lane[:, 2], uv=False)
            gt_lane = np.stack([x_ego, y_ego, z_ego], axis=-1)
            gt_lanes.append(gt_lane)
        lane_points = gt_lanes
        
        if len(lane_points) > 0:
            img_mask = np.zeros((len(lane_points), 2, self.img_h, self.img_w), dtype=np.float32)
            bev_mask = np.zeros((len(lane_points), 2, self.bev_h, self.bev_w), dtype=np.float32)
            for i, gt_lane in enumerate(lane_points):                
                # draw img mask
                x_img, y_img, z_img = self.projective_transformation(
                    P_ego2img, gt_lane[:, 0], gt_lane[:, 1], gt_lane[:, 2], uv=True)
                points = np.stack([x_img, y_img, z_img], axis=1)
                points = self.lane_interpolate(points)

                points[:, :2] = points[:, :2].round()
                points = points[(points[:, 0] >= 0) & (points[:, 0] < self.img_w) &
                                (points[:, 1] >= 0) & (points[:, 1] < self.img_h)]
                points_xs = points[:, 0].astype(np.int64)
                points_ys = points[:, 1].astype(np.int64)
                points_zs = points[:, 2] - depth_prior[points_ys, points_xs]
                img_mask[i, 0, points_ys, points_xs] = 1
                img_mask[i, 1, points_ys, points_xs] = points_zs
                
                # draw bev mask
                x_bev, y_bev, z_bev = self.projective_transformation(
                    self.P_ego2bev, gt_lane[:, 0], gt_lane[:, 1], gt_lane[:, 2], uv=False)
                points = np.stack([x_bev, y_bev, z_bev], axis=1)
                points = self.lane_interpolate(points)
                
                points[:, :2] = points[:, :2].round()
                points = points[(points[:, 0] >= 0) & (points[:, 0] < self.bev_w) &
                                (points[:, 1] >= 0) & (points[:, 1] < self.bev_h)]
                points_xs = points[:, 0].astype(np.int64)
                points_ys = points[:, 1].astype(np.int64)
                points_zs = points[:, 2]
                bev_mask[i, 0, points_ys, points_xs] = 1
                bev_mask[i, 1, points_ys, points_xs] = points_zs

            img_mask = img_mask.reshape(-1, self.img_h, self.img_w).transpose((1, 2, 0))
            bev_mask = bev_mask.reshape(-1, self.bev_h, self.bev_w).transpose((1, 2, 0))
            results['img_mask'] = img_mask
            results['bev_mask'] = bev_mask

        if 'lane_attris' in results:
            lane_attris = results['lane_attris']
            assert len(lane_points) == len(lane_attris), \
                'Number of lanes {} and number of attributes {} are different'.format(len(lane_points), len(lane_attris))
            results['lane_attris'] = np.array(lane_attris, dtype=np.int64)

        img = cv2.warpPerspective(img_org, H_img2img, (self.img_w, self.img_h), flags=cv2.INTER_LINEAR)
        results['img'] = img
        results['bev_shape'] = (self.bev_h, self.bev_w)
        results['lane_points'] = lane_points
        results['depth_prior'] = depth_prior
        results['calibration'] = {}
        results['calibration']['trans_ego2cam'] = P_ego2cam
        results['calibration']['trans_cam2ego'] = P_cam2ego
        results['calibration']['trans_cam2img'] = P_cam2img
        results['calibration']['trans_ego2img'] = P_ego2img
        results['calibration']['trans_ego2bev'] = self.P_ego2bev
        results['calibration']['trans_bev2ego'] = self.P_bev2ego
        return results


# class GenerateLaneLine3D(object):
#
#     def __init__(self, img_h, img_w, ipm_h, ipm_w, top_view_region, cut_height=0, thickness=1, data_aug=False):
#         self.img_w, self.img_h = img_w, img_h
#         self.ipm_w, self.ipm_h = ipm_w, ipm_h
#
#         # np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
#         self.top_view_region = top_view_region
#         self.x_min, self.x_max = self.top_view_region[0, 0], self.top_view_region[1, 0]
#         self.y_min, self.y_max = self.top_view_region[2, 1], self.top_view_region[0, 1]
#
#         self.cut_height = cut_height
#         self.thickness = thickness
#         self.data_aug = data_aug
#
#         self.H_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
#                                                                [self.ipm_w-1, 0],
#                                                                [0, self.ipm_h-1],
#                                                                [self.ipm_w-1, self.ipm_h-1]]),
#                                                    np.float32(self.top_view_region))
#         self.H_g2ipm = np.linalg.inv(self.H_ipm2g)
#
#     @staticmethod
#     def homograpthy_g2im(cam_pitch, cam_height, K):
#         # transform top-view region to original image region
#         R_g2c = np.array([[1, 0, 0],
#                           [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
#                           [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
#         H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))
#         return H_g2im
#
#     @staticmethod
#     def projection_g2im(cam_pitch, cam_height, K):
#         P_g2c = np.array([[1, 0, 0, 0],
#                           [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
#                           [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch), 0]])
#         P_g2im = np.matmul(K, P_g2c)
#         return P_g2im
#
#     @staticmethod
#     def homograpthy_g2im_extrinsic(E, K):
#         """E: extrinsic matrix, 4*4"""
#         E_inv = np.linalg.inv(E)[0:3, :]
#         H_g2c = E_inv[:, [0, 1, 3]]
#         H_g2im = np.matmul(K, H_g2c)
#         return H_g2im
#
#     @staticmethod
#     def projection_g2im_extrinsic(E, K):
#         E_inv = np.linalg.inv(E)[0:3, :]
#         P_g2im = np.matmul(K, E_inv)
#         return P_g2im
#
#     @staticmethod
#     def homography_crop_resize(org_img_size, crop_y, resize_img_size):
#         """
#             compute the homography matrix transform original image to cropped and resized image
#         :param org_img_size: [org_h, org_w]
#         :param crop_y:
#         :param resize_img_size: [resize_h, resize_w]
#         :return:
#         """
#         # transform original image region to network input region
#         ratio_x = resize_img_size[1] / org_img_size[1]
#         ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
#         H_c = np.array([[ratio_x, 0, 0],
#                         [0, ratio_y, -ratio_y * crop_y],
#                         [0, 0, 1]])
#         return H_c
#
#     @staticmethod
#     def homographic_transformation(Matrix, x, y):
#         """
#         Helper function to transform coordinates defined by transformation matrix
#
#         Args:
#                 Matrix (multi dim - array): 3x3 homography matrix
#                 x (array): original x coordinates
#                 y (array): original y coordinates
#         """
#         ones = np.ones((1, len(y)))
#         coordinates = np.vstack((x, y, ones))
#         trans = np.matmul(Matrix, coordinates)
#
#         x_vals = trans[0, :] / trans[2, :]
#         y_vals = trans[1, :] / trans[2, :]
#         return x_vals, y_vals
#
#     @staticmethod
#     def projective_transformation(Matrix, x, y, z):
#         """
#         Helper function to transform coordinates defined by transformation matrix
#
#         Args:
#                 Matrix (multi dim - array): 3x4 projection matrix
#                 x (array): original x coordinates
#                 y (array): original y coordinates
#                 z (array): original z coordinates
#         """
#         ones = np.ones((1, len(z)))
#         coordinates = np.vstack((x, y, z, ones))
#         trans = np.matmul(Matrix, coordinates)
#
#         x_vals = trans[0, :] / trans[2, :]
#         y_vals = trans[1, :] / trans[2, :]
#         return x_vals, y_vals
#
#     @staticmethod
#     def make_lane_y_mono_inc(lane):
#         """
#             Due to lose of height dim, projected lanes to flat ground plane may not have monotonically increasing y.
#             This function trace the y with monotonically increasing y, and output a pruned lane
#         :param lane:
#         :return:
#         """
#         idx2del = []
#         max_y = lane[0, 1]
#         for i in range(1, lane.shape[0]):
#             # hard-coded a smallest step, so the far-away near horizontal tail can be pruned
#             if lane[i, 1] <= max_y + 3:
#                 idx2del.append(i)
#             else:
#                 max_y = lane[i, 1]
#         lane = np.delete(lane, idx2del, 0)
#         return lane
#
#     def convert_lanes_3d_to_gflat(self, lanes, P_g2gflat):
#         """
#             Convert a set of lanes from 3D ground coordinates [X, Y, Z], to IPM-based
#             flat ground coordinates [x_gflat, y_gflat, Z]
#         :param lanes: a list of N x 3 numpy arrays recording a set of 3d lanes
#         :param P_g2gflat: projection matrix from 3D ground coordinates to flat ground coordinates
#         :return:
#         """
#         # TODO: this function can be simplified with the derived formula
#         lanes_gflat = []
#         for lane in lanes:
#             # convert gt label to anchor label
#             lane_gflat_x, lane_gflat_y = self.projective_transformation(P_g2gflat, lane[:, 0], lane[:, 1], lane[:, 2])
#             lane_gflat = np.stack([lane_gflat_x, lane_gflat_y, lane[:, 2]], axis=1)
#             lanes_gflat.append(lane_gflat)
#         return lanes_gflat
#
#     def transform_mats_impl(self, cam_extrinsics, cam_intrinsics, cam_pitch=None, cam_height=None, img_shape=None):
#         if cam_extrinsics is not None:
#             H_g2im = self.homograpthy_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
#             P_g2im = self.projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
#         else:
#             H_g2im = self.homograpthy_g2im(cam_pitch, cam_height, cam_intrinsics)
#             P_g2im = self.projection_g2im(cam_pitch, cam_height, cam_intrinsics)
#
#         H_crop = self.homography_crop_resize([img_shape[0], img_shape[1]], self.cut_height, [self.img_h, self.img_w])
#         H_im2ipm = np.linalg.inv(np.matmul(H_crop, np.matmul(H_g2im, self.H_ipm2g)))
#         return H_g2im, P_g2im, H_crop, H_im2ipm
#
#     @staticmethod
#     def data_aug_rotate(img):
#         # assume img in PIL image format
#         rot = random.uniform(-np.pi / 18, np.pi / 18)
#         # rot = random.uniform(-10, 10)
#         center_x = img.shape[1] / 2
#         center_y = img.shape[0] / 2
#         rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
#         img_rot = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
#         # img_rot = img.rotate(rot)
#         # rot = rot / 180 * np.pi
#         rot_mat = np.vstack([rot_mat, [0, 0, 1]])
#         return img_rot, rot_mat
#
#     def __call__(self, results):
#         cam_intrinsics = np.array(results['calibration']['intrinsics'])
#         cam_extrinsics = np.array(results['calibration']['extrinsics'])
#         gt_lanes = [np.array(lane) for lane in results['lane_points']]
#         gt_category = results['lane_attris']
#         img_org = results['img']
#
#         P_g2im = self.projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
#         H_g2im = self.homograpthy_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
#         H_im2g = np.linalg.inv(H_g2im)
#         P_g2gflat = np.matmul(H_im2g, P_g2im)
#         gt_lanes_3d = self.convert_lanes_3d_to_gflat(gt_lanes, P_g2gflat)
#
#         gt_lanes_ = []
#         gt_category_ = []
#         for i in range(len(gt_lanes_3d)):
#             gt_lane_3d = gt_lanes_3d[i]
#             # prune out points not in valid range, requires additional points to interpolate better
#             # prune out-of-range points after transforming to flat ground space, update visibility vector
#             valid_indices = np.logical_and(np.logical_and(gt_lane_3d[:, 1] > 0, gt_lane_3d[:, 1] < 200),
#                                            np.logical_and(gt_lane_3d[:, 0] > 3 * self.x_min,
#                                                           gt_lane_3d[:, 0] < 3 * self.x_max))
#             gt_lane_3d = gt_lane_3d[valid_indices, ...]
#             # use more restricted range to determine deletion or not
#             if gt_lane_3d.shape[0] < 2 or np.sum(np.logical_and(gt_lane_3d[:, 0] > self.x_min,
#                                                                 gt_lane_3d[:, 0] < self.x_max)) < 2:
#                 continue
#             # only keep the portion y is monotonically increasing above a threshold, to prune those super close points
#             gt_lane_3d = self.make_lane_y_mono_inc(gt_lane_3d)
#             if gt_lane_3d.shape[0] < 2:
#                 continue
#             gt_lanes_.append(gt_lane_3d)
#             gt_category_.append(gt_category[i])
#         gt_lanes_3d = gt_lanes_
#         gt_category_3d = gt_category_
#
#         img = F.crop(img_org, self.cut_height, 0, img_org.shape[0]-self.cut_height, img_org.shape[1])
#         img = F.resize(img, size=(self.img_h, self.img_w), interpolation=InterpolationMode.BILINEAR)
#
#         if self.data_aug:
#             img, aug_mat = self.data_aug_rotate(img)
#
#         assert len(gt_lanes) == len(gt_category)
#
#         # prepare binary segmentation label map
#         img_mask = np.zeros((2, self.img_h, self.img_w), dtype=np.float32)
#         for i, lane in enumerate(gt_lanes):
#             H_g2im, P_g2im, H_crop, H_im2ipm = \
#                 self.transform_mats_impl(cam_extrinsics, cam_intrinsics, img_shape=img_org.shape)
#             M = np.matmul(H_crop, P_g2im)
#             # update transformation with image augmentation
#             if self.data_aug:
#                 M = np.matmul(aug_mat, M)
#             x_2d, y_2d = self.projective_transformation(M, lane[:, 0], lane[:, 1], lane[:, 2])
#             for j in range(len(x_2d) - 1):
#                 img_mask[0] = cv2.line(img_mask[0],
#                                        (int(x_2d[j]), int(y_2d[j])), (int(x_2d[j+1]), int(y_2d[j+1])),
#                                        color=(i + 1,), thickness=self.thickness)
#                 img_mask[1] = cv2.line(img_mask[1],
#                                        (int(x_2d[j]), int(y_2d[j])), (int(x_2d[j+1]), int(y_2d[j+1])),
#                                        color=(gt_category[i],), thickness=self.thickness)
#         img_mask = img_mask.transpose((1, 2, 0))
#
#         map_mask = np.zeros((3, self.ipm_h, self.ipm_w), dtype=np.float32)
#         for i, gt_lane_3d in enumerate(gt_lanes_3d):
#             x_g = gt_lane_3d[:, 0]
#             y_g = gt_lane_3d[:, 1]
#             z_g = gt_lane_3d[:, 2]
#             x_ipm, y_ipm = self.homographic_transformation(self.H_g2ipm, x_g, y_g)
#
#             points = []
#             for j, (p_curr, p_next) in enumerate(zip(gt_lane_3d[:-1], gt_lane_3d[1:])):
#                 x_max, x_min = max(x_ipm[j], x_ipm[j + 1]), min(x_ipm[j], x_ipm[j + 1])
#                 y_max, y_min = max(y_ipm[j], y_ipm[j + 1]), min(y_ipm[j], y_ipm[j + 1])
#                 inter_num = int(np.sqrt((x_max - x_min + 1) ** 2 + (y_max - y_min + 1) ** 2)) + 1
#                 point_new = np.linspace(p_curr, p_next, num=inter_num, endpoint=True)
#                 points.extend(list(point_new))
#             points = np.array(points).reshape(-1, 3)
#
#             x_g, y_g, z_g = points.T
#             x_ipm, y_ipm = self.homographic_transformation(self.H_g2ipm, x_g, y_g)
#             points = np.stack([x_ipm, y_ipm, z_g], axis=1)
#
#             points[:, :2] = points[:, :2].round()
#             points = points[(points[:, 0] >= 0) & (points[:, 0] < self.ipm_w) &
#                             (points[:, 1] >= 0) & (points[:, 1] < self.ipm_h)]
#             points_xs = points[:, 0].astype(np.int64)
#             points_ys = points[:, 1].astype(np.int64)
#             points_zs = points[:, 2]
#             map_mask[0, points_ys, points_xs] = i + 1
#             map_mask[1, points_ys, points_xs] = points_zs
#             map_mask[2, points_ys, points_xs] = gt_category_3d[i]
#         map_mask = map_mask.transpose((1, 2, 0))
#
#         org_h, org_w = img_org.shape[:2]
#         new_h, new_w = img.shape[:2]
#         w_scale = new_w / org_w
#         h_scale = new_h / org_h
#         scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
#
#         results['img'] = img
#         results['img_mask'] = img_mask
#         results['map_mask'] = map_mask
#         results['lane_points'] = gt_lanes
#         results['img_shape'] = img.shape
#         results['pad_shape'] = img.shape
#         results['ori_shape'] = img_org.shape
#         results['image_offset'] = np.array([0, self.cut_height])
#         results['scale_factor'] = scale_factor
#         return results


# class FixedIMGBEVTrans(object):
#
#     def __init__(
#             self,
#             img_height=1080,
#             img_width=1920,
#             new_height=768,
#             new_width=1344,
#             camera_ids=[2, 3, 5, 10, 11, 12, 13, 14, 15],
#             cam_calibration_path="s3://zhuqiong-oss-hhb/data/LaneDetection/CameraCalibration_0309_upload",
#             virtual_cam_id=15,
#             virtual_cam_calibration_path="s3://zhuqiong-oss-hhb/data/LaneDetection/CameraCalibration_0309_upload",
#             virtual_height=1080,
#             virtual_width=1920,
#             use_virtual_cam=False
#     ):
#         self.img_height = img_height
#         self.img_width = img_width
#         self.new_height = new_height
#         self.new_width = new_width
#         self.cam_calibration_path = cam_calibration_path
#         self.camera_ids = camera_ids
#         self.virtual_cam_id = virtual_cam_id
#         self.virtual_cam_calibration_path = virtual_cam_calibration_path
#         self.virtual_height = virtual_height
#         self.virtual_width = virtual_width
#         self.use_virtual_cam = use_virtual_cam
#
#         self.camera_params = {}
#         self.virtual_camera_params = {}
#
#         # load camera params
#         self.load_camera_params()
#
#     def load_camera_params(self):
#         """ Load camera calibration parameters only when initialization """
#         for cam_id in self.camera_ids:
#             self.camera_params[cam_id] = {}
#
#             root_path = self.cam_calibration_path
#             cam_intrinsic_param_path = os.path.join(root_path, "camera_params",
#                                                     "camera_{}_intrinsic.json".format(cam_id))
#             lidar2cam_extrinsic_param_path = os.path.join(root_path, "camera_params",
#                                                           "camera_{}_extrinsic.json".format(cam_id))
#             lidar2ego_param_path = os.path.join(root_path, "lidar_params", "lidar_ego.json")
#
#             with smart_open(cam_intrinsic_param_path, "r") as f:
#                 cam_intrinsic_params = json.load(f)
#             with smart_open(lidar2cam_extrinsic_param_path, "r") as f:
#                 lidar2cam_extrinsic_params = json.load(f)
#             with smart_open(lidar2ego_param_path, "r") as f:
#                 lidar2ego_params = json.load(f)
#
#             self.camera_params[cam_id]['lidar2cam'] = lidar2cam_extrinsic_params
#             self.camera_params[cam_id]['lidar2ego'] = lidar2ego_params
#
#             mode = cam_intrinsic_params['distortion_model']
#             K = np.array(cam_intrinsic_params['K'], dtype=np.float32).reshape(3, 3)
#             D = np.array(cam_intrinsic_params['D'], dtype=np.float32)
#             self.camera_params[cam_id]['K'] = K
#             self.camera_params[cam_id]['D'] = D
#             self.camera_params[cam_id]['mode'] = mode
#
#             # store undistort mapx,mapy
#             if mode == 'pinhole':
#                 mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, (self.img_width, self.img_height), 5)
#                 self.camera_params[cam_id]['mapx'] = mapx
#                 self.camera_params[cam_id]['mapy'] = mapy
#             elif mode == 'fisheye':
#                 mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, None, K, (self.img_width, self.img_height), 5)
#                 self.camera_params[cam_id]['mapx'] = mapx
#                 self.camera_params[cam_id]['mapy'] = mapy
#
#         if self.use_virtual_cam:
#             cam_id = self.virtual_cam_id
#             self.virtual_camera_params[cam_id] = {}
#
#             root_path = self.virtual_cam_calibration_path
#             cam_intrinsic_param_path = os.path.join(root_path, "camera_params",
#                                                     "camera_{}_intrinsic.json".format(cam_id))
#             lidar2cam_extrinsic_param_path = os.path.join(root_path, "camera_params",
#                                                           "camera_{}_extrinsic.json".format(cam_id))
#             lidar2ego_param_path = os.path.join(root_path, "lidar_params", "lidar_ego.json")
#
#             with smart_open(cam_intrinsic_param_path, "r") as f:
#                 cam_intrinsic_params = json.load(f)
#             with smart_open(lidar2cam_extrinsic_param_path, "r") as f:
#                 lidar2cam_extrinsic_params = json.load(f)
#             with smart_open(lidar2ego_param_path, "r") as f:
#                 lidar2ego_params = json.load(f)
#
#             self.virtual_camera_params[cam_id]['lidar2cam'] = lidar2cam_extrinsic_params
#             self.virtual_camera_params[cam_id]['lidar2ego'] = lidar2ego_params
#
#             mode = cam_intrinsic_params['distortion_model']
#             K = np.array(cam_intrinsic_params['K'], dtype=np.float32).reshape(3, 3)
#             D = np.array(cam_intrinsic_params['D'], dtype=np.float32)
#             self.virtual_camera_params[cam_id]['K'] = K
#             self.virtual_camera_params[cam_id]['D'] = D
#             self.virtual_camera_params[cam_id]['mode'] = mode
#
#     def undistort_img(self, img, cam_id):
#         """
#         Undistort img &lane points using stored camera params
#         """
#         mapx = self.camera_params[cam_id]['mapx']
#         mapy = self.camera_params[cam_id]['mapy']
#         undistort_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
#         return undistort_img
#
#     @staticmethod
#     def transform_matrix(
#             translation: np.ndarray = np.array([0, 0, 0]),
#             rotation: Quaternion = Quaternion([1, 0, 0, 0]),
#             inverse: bool = False,
#     ) -> np.ndarray:
#         """
#         Convert pose to transformation matrix.
#         :param translation: <np.float32: 3>. Translation in x, y, z.
#         :param rotation: Rotation in quaternions (w ri rj rk).
#         :param inverse: Whether to compute inverse transform matrix.
#         :return: <np.float32: 4, 4>. Transformation matrix.
#         """
#         tm = np.eye(4)
#         if inverse:
#             rot_inv = rotation.rotation_matrix.T
#             trans = np.transpose(-np.array(translation))
#             tm[:3, :3] = rot_inv
#             tm[:3, 3] = rot_inv.dot(trans)
#         else:
#             tm[:3, :3] = rotation.rotation_matrix
#             tm[:3, 3] = np.transpose(np.array(translation))
#         return tm
#
#     def get_ego2cam_params_v1(self, camera_params, cam_id=15):
#         K         = camera_params[cam_id]['K']
#         D         = camera_params[cam_id]['D']
#         model     = camera_params[cam_id]['mode']
#         lidar2cam = camera_params[cam_id]['lidar2cam']
#         lidar2ego = camera_params[cam_id]['lidar2ego']
#
#         trans_ego2lidar = self.transform_matrix(
#             np.array(list(lidar2ego['transform']['translation'].values())),
#             Quaternion(list(lidar2ego['transform']['rotation'].values())), inverse=True)
#
#         trans_lidar2cam = self.transform_matrix(
#             np.array(list(lidar2cam['transform']['translation'].values())),
#             Quaternion(list(lidar2cam['transform']['rotation'].values())), inverse=False)
#
#         trans_ego2cam = trans_lidar2cam @ trans_ego2lidar
#         R_ego2cam = trans_ego2cam[:3, :3]
#         T_ego2cam = trans_ego2cam[:3, 3]
#         return R_ego2cam, T_ego2cam, K, D, model
#
#     def get_ego2cam_params_v2(self, calibration):
#         cam_intrinsic_params = calibration['cam_intrinsic']
#         lidar2cam = calibration['cam_extrinsic']
#         lidar2ego = calibration['lidar2ego']
#
#         K = np.array(cam_intrinsic_params['K']).reshape(3, 3)
#         D = np.array(cam_intrinsic_params['D'])
#         model = cam_intrinsic_params['distortion_model']
#
#         trans_ego2lidar = self.transform_matrix(
#             np.array(list(lidar2ego['translation'].values())),
#             Quaternion(list(lidar2ego['rotation'].values())), inverse=True)
#
#         trans_lidar2cam = self.transform_matrix(
#             np.array(list(lidar2cam['translation'].values())),
#             Quaternion(list(lidar2cam['rotation'].values())), inverse=False)
#
#         trans_ego2cam = trans_lidar2cam @ trans_ego2lidar
#         R_ego2cam = trans_ego2cam[:3, :3]
#         T_ego2cam = trans_ego2cam[:3, 3]
#         return R_ego2cam, T_ego2cam, K, D, model
#
#     def __call__(self, results):
#         img    = results['img']
#         cam_id = results['cam_id']
#
#         img = self.undistort_img(img, cam_id)
#
#         if 'calibration' in results:
#             R_e2c, T_e2c, K, D, model = self.get_ego2cam_params_v2(results['calibration'])
#         else:
#             R_e2c, T_e2c, K, D, model = self.get_ego2cam_params_v1(self.camera_params, cam_id)
#             results['calibration'] = {}
#
#         if self.use_virtual_cam:
#             virtual_R_e2c, virtual_T_e2c, virtual_K, virtual_D, virtual_model = \
#                 self.get_ego2cam_params_v1(self.virtual_camera_params, self.virtual_cam_id)
#             trans_img2img_prime = virtual_K @ np.linalg.inv(K)
#             h, w = self.virtual_height, self.virtual_width
#             R_e2c, T_e2c = virtual_R_e2c, virtual_T_e2c
#         else:
#             h, w = self.img_height, self.img_width
#             trans_img2img_prime = np.eye(3)
#
#         new_h, new_w = self.new_height, self.new_width
#         w_scale = new_w / w
#         h_scale = new_h / h
#         img_shape = (new_h, new_w, 3)
#         pad_shape = (new_h, new_w, 3)
#         scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
#
#         rescale_mat = np.eye(3)
#         rescale_mat[0, 0] = w_scale
#         rescale_mat[1, 1] = h_scale
#         trans_img2img_prime = rescale_mat @ trans_img2img_prime
#
#         results['img_shape'] = img_shape
#         results['pad_shape'] = pad_shape
#         results['scale_factor'] = scale_factor
#
#         trans_cam2img = K
#         trans_cam2img = trans_img2img_prime @ trans_cam2img
#
#         trans_ego2cam = np.eye(4)
#         trans_ego2cam[:3, :3] = R_e2c
#         trans_ego2cam[:3, 3] = T_e2c
#         trans_cam2ego = np.linalg.inv(trans_ego2cam)
#
#         R_ego2cam = trans_ego2cam[:3, :3]
#         T_ego2cam = trans_ego2cam[:3, 3]
#
#         # transform matrix from ego coord --> img pixels
#         trans_ego2img = np.eye(4)
#         trans_ego2img[:3, :3] = trans_cam2img @ R_ego2cam
#         trans_ego2img[:3, 3] = trans_cam2img @ T_ego2cam.T
#
#         results['calibration']['trans_ego2cam'] = trans_ego2cam
#         results['calibration']['trans_cam2ego'] = trans_cam2ego
#         results['calibration']['trans_cam2img'] = trans_cam2img
#         results['calibration']['trans_ego2img'] = trans_ego2img
#
#         img = cv2.warpPerspective(img, trans_img2img_prime, (new_w, new_h), flags=cv2.INTER_LINEAR)
#
#         # output-imgs should be numpy-array
#         results['img'] = img
#         results['ori_shape'] = (h, w, 3)
#         return results
