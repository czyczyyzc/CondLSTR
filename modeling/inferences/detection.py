import os
import cv2
import mmcv
import torch
import numpy as np
import os.path as osp


class DetectionInference(object):

    def __init__(self, visualize=False, root=None):
        self.visualize = visualize
        self.root = root
        os.makedirs(root, exist_ok=True)

    def __call__(self, pred_dict, data_dict):
        box_preds = pred_dict['bboxes']
        scr_preds = pred_dict['scores']
        cls_preds = pred_dict['classes']
        img_metas = data_dict['img_metas']

        pred_list = []
        for i, (img_meta, box_pred, scr_pred, cls_pred) in \
                enumerate(zip(img_metas, box_preds, scr_preds, cls_preds)):
            img_size = img_meta['img_shape']
            ori_size = img_meta['ori_shape']
            img_name = img_meta['filename']
            im_ratio = img_meta['scale_factor']
            im_offset = img_meta['image_offset'] if 'image_offset' in img_meta else np.array([0, 0])

            box_pred = box_pred.reshape(-1, 2)
            box_pred = box_pred / im_ratio[:2] + im_offset[:2]
            box_pred = box_pred.reshape(-1, 4)

            box_pred = box_pred.tolist()
            scr_pred = scr_pred.tolist()
            cls_pred = cls_pred.tolist()

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

                for box, score, category in zip(box_pred, scr_pred, cls_pred):
                    if score < 0.6:
                        continue
                    pt1 = (int(box[0]), int(box[1]))
                    pt2 = (int(box[2]), int(box[3]))
                    img_data = cv2.rectangle(img_data, pt1, pt2, color=(0, 255, 0), thickness=4)
                    img_data = cv2.putText(img_data, '{:.2f}'.format(score), pt1, cv2.FONT_HERSHEY_SIMPLEX,
                                           0.75, (255, 0, 0), 2)
                cv2.imwrite(img_path, img_data)

            res_dict = {
                'image_name': osp.basename(img_name),
                'bboxes': box_pred,
                'scores': scr_pred,
                'classes': cls_pred,
            }
            pred_list.append(res_dict)
        return pred_list
