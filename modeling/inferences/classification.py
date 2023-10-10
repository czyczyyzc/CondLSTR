import os
import cv2
import mmcv
import torch
import numpy as np
import os.path as osp
from .utils import obtain_colors


class ClassificationInference(object):

    def __init__(self, max_instances=512, visualize=True, root=None):
        self.visualize = visualize
        self.root = root
        self.colors = obtain_colors(max_instances, shuffle=True)
        os.makedirs(root, exist_ok=True)

    def __call__(self, pred_dict, data_dict):
        img_metas = data_dict['img_metas']
        scores = pred_dict['score']
        classes = pred_dict['class']
        
        scores = scores.tolist() if isinstance(scores, np.ndarray) else scores
        classes = classes.tolist() if isinstance(classes, np.ndarray) else classes

        pred_list = []
        for i, (score, category, img_meta) in enumerate(zip(scores, classes, img_metas)):
            img_size = img_meta['img_shape']
            ori_size = img_meta['ori_shape']
            img_name = img_meta['filename']
            
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
                
                if not isinstance(category, str):
                    category = img_meta['class_dict'][category] if 'class_dict' in img_meta else str(category)
                cv2.putText(img_data, category, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imwrite(img_path, img_data)

            res_dict = {
                'image_name': osp.basename(img_name),
                'score': score,
                'class': category,
            }
            pred_list.append(res_dict)
        return pred_list
