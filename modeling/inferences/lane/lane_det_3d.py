import os
import cv2
import mmcv
import torch
import os.path as osp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ..utils import obtain_colors

mpl.rcParams['legend.fontsize'] = 10


class LaneDet3DInference(object):

    def __init__(self, max_lanes=20, visualize=False, root=None):
        self.visualize = visualize
        self.root = root
        self.colors = obtain_colors(max_lanes, shuffle=True)
        os.makedirs(root, exist_ok=True)

    def __call__(self, pred_dict, data_dict):
        """
        seg_pred: (N, 3, H, W)
        """
        lane_points = pred_dict['lane_points']
        lane_scores = pred_dict['lane_scores']
        lane_attris = pred_dict['lane_attris'] if 'lane_attris' in pred_dict else [None] * len(lane_points)
        img_metas = data_dict['img_metas']

        pred_list = []
        for i, (img_meta, points_list, scores_list, attris_list) in \
                enumerate(zip(img_metas, lane_points, lane_scores, lane_attris)):
            img_size = img_meta['img_shape']
            ori_size = img_meta['ori_shape']
            img_name = img_meta['filename']
            b2e_tran = img_meta['calibration']['trans_bev2ego']
            e2i_tran = img_meta['calibration']['trans_ego2img']
            img_ratio = img_meta['scale_factor']
            img_offset = img_meta['image_offset'] if 'image_offset' in img_meta else np.array([0, 0])
            
            points_2d_list = []
            points_3d_list = []
            for points_3d in points_list:
                points_3d = points_3d[0] if isinstance(points_3d, tuple) else points_3d
                points_3d = np.asarray(points_3d)
                points_3d = np.concatenate([points_3d, np.ones_like(points_3d[:, 0:1])], axis=1).T
                points_3d = np.matmul(b2e_tran, points_3d)
                points_2d = np.matmul(e2i_tran, points_3d)
                points_2d = points_2d[:2] / points_2d[2:3]
                points_3d = points_3d.T
                points_2d = points_2d.T
                points_2d = points_2d / img_ratio[:2] + img_offset[:2]
                points_2d = points_2d.astype(np.float64)
                points_2d = points_2d.tolist()
                points_3d = points_3d[:, :3]
                points_3d = points_3d.astype(np.float64)
                points_3d = points_3d.tolist()
                points_2d_list.append(points_2d)
                points_3d_list.append(points_3d)
            
            attris_list = [None] * len(points_list) if attris_list is None else attris_list
            scores_list = scores_list.tolist() if isinstance(scores_list, np.ndarray) else scores_list
            attris_list = attris_list.tolist() if isinstance(attris_list, np.ndarray) else attris_list

            if self.visualize:
                img_path = osp.join(self.root, osp.splitext(osp.basename(img_name))[0] + '.jpg')
                img_data = data_dict['img'][i]
                if isinstance(img_data, torch.Tensor):
                    img_data = img_data.permute(1, 2, 0).cpu().numpy()    # (H, W, C)
                if 'img_norm_cfg' in img_meta:
                    img_mean = img_meta['img_norm_cfg']['mean']
                    img_std = img_meta['img_norm_cfg']['std']
                    to_rgb = img_meta['img_norm_cfg']['to_rgb']
                    img_data = img_data * img_std + img_mean
                    if to_rgb:
                        img_data = img_data[:, :, ::-1]
                img_data = img_data.astype(np.uint8)
                img_data = img_data[:img_size[0], :img_size[1]]
                img_data = mmcv.imresize(img_data, (ori_size[1], ori_size[0]), return_scale=False,
                                         interpolation='bilinear', backend='cv2')

                fig = plt.figure()  # plt.figure(figsize=(10,6), dpi=100)
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim(-10, 10)
                ax.set_ylim(3, 103)
                ax.set_zlim(-5, 5)

                for j, (points_2d, points_3d, attri, score) in enumerate(zip(points_2d_list, points_3d_list, attris_list, scores_list)):
                    for p_curr, p_next in zip(points_2d[:-1], points_2d[1:]):
                        pt1 = (int(round(p_curr[0])), int(round(p_curr[1])))
                        pt2 = (int(round(p_next[0])), int(round(p_next[1])))
                        rgb = tuple(map(int, list(self.colors[j])))
                        img_data = cv2.line(img_data, pt1, pt2, color=rgb, thickness=4)
                    
                    if attri is not None:
                        if not isinstance(attri, str):
                            attri = img_meta['class_dict'][attri] if 'class_dict' in img_meta else str(attri)
                        center = points_2d[2 * len(points_2d) // 3]
                        center = (int(round(center[0]) - img_offset[0]), int(round(center[1]) - img_offset[1]))
                        img_data = cv2.putText(img_data, attri, center, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    
                    ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], label='')
                
                # 显示图例
                ax.legend()
                # 显示图形
                plt.savefig(img_path.replace('.jpg', '_3d.jpg'))
                plt.close()
                cv2.imwrite(img_path, img_data)

            res_dict = {
                'image_name': osp.basename(img_name),
                'points_2d': points_2d_list,
                'points_3d': points_3d_list,
                'attris': attris_list,
                'scores': scores_list,
            }
            pred_list.append(res_dict)
        return
