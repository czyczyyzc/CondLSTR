import os
import cv2
import mmcv
import torch
import numpy as np
import os.path as osp
from pycocotools import mask as mask_utils
from .utils import obtain_colors


class InstanceSegInference(object):

    def __init__(self, max_instances=512, visualize=True, root=None):
        self.visualize = visualize
        self.root = root
        self.colors = obtain_colors(max_instances, shuffle=True)
        os.makedirs(root, exist_ok=True)

    def __call__(self, pred_dict, data_dict):
        img_metas = data_dict['img_metas']
        mask_list = pred_dict['masks']
        bbox_list = pred_dict['bboxes']
        score_list = pred_dict['scores']
        class_list = pred_dict['classes'] if 'classes' in pred_dict else [None] * len(img_metas)

        pred_list = []
        for i, (masks, bboxes, scores, classes, img_meta) in \
                enumerate(zip(mask_list, bbox_list, score_list, class_list, img_metas)):
            img_size = img_meta['img_shape']
            ori_size = img_meta['ori_shape']
            img_name = img_meta['filename']
            img_ratio = img_meta['scale_factor']
            img_offset = img_meta['image_offset'] if 'image_offset' in img_meta else np.array([0, 0])
            
            bboxes = np.asarray(bboxes).reshape(-1, 2) / img_ratio[:2] + img_offset[:2]
            bboxes = bboxes.reshape(-1, 4).tolist()
            scores = scores.tolist() if isinstance(scores, np.ndarray) else scores
            classes = classes.tolist() if isinstance(classes, np.ndarray) else classes
            
            if len(masks) > 0:
                # 如果 mask 是 RLE 格式的，需要先解码为二进制格式
                if isinstance(masks, list):
                    if isinstance(masks[0], dict):
                        masks = mask_utils.decode(masks)
                    elif isinstance(masks[0], np.ndarray):
                        masks = np.stack(masks, axis=2)
                    else:
                        raise NotImplementedError
                elif isinstance(masks, torch.Tensor):
                    masks = masks.permute(1, 2, 0).cpu().numpy()
                elif isinstance(masks, np.ndarray):
                    masks = masks.transpose((1, 2, 0))
                else:
                    raise NotImplementedError
                masks = masks[:img_size[0], :img_size[1]]
                masks = mmcv.imresize(masks, (ori_size[1], ori_size[0]), return_scale=False, 
                                        interpolation='nearest', backend='cv2')
                masks = masks[:, :, None] if len(masks.shape) == 2 else masks
                masks = masks.transpose((2, 0, 1))
            else:
                masks = np.zeros((0, ori_size[0], ori_size[1]), dtype=np.uint8)
            
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
                img_data = self.draw_masks_on_image(img_data, masks)
                img_data = self.draw_boxes_on_image(img_data, bboxes, classes, scores, img_meta)
                cv2.imwrite(img_path, img_data)

            # 将二进制格式的 mask 转换为 RLE 格式
            masks = masks.transpose((1, 2, 0))
            masks = mask_utils.encode(np.asfortranarray(masks))

            res_dict = {
                'image_name': osp.basename(img_name),
                'masks': masks,
                'bboxes': bboxes,
                'scores': scores,
            }
            if classes is not None:
                res_dict['classes'] = classes
            pred_list.append(res_dict)
        return pred_list

    def draw_masks_on_image(self, image, masks, alpha=0.5):
        # 用于保存叠加了mask的图像
        image_with_masks = image.copy()
        for i, mask in enumerate(masks):
            
            assert mask.shape[:2] == image_with_masks.shape[:2], \
                "mask shape {} and image shape {} don't match".format(mask.shape, image_with_masks.shape)

            # 将 mask 转换为三通道，用随机颜色填充
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            colored_mask[mask > 0] = self.colors[i]

            # 将 mask 画到图像上
            image_with_masks = cv2.addWeighted(image_with_masks, 1.0, colored_mask, alpha, 0)
        return image_with_masks

    @staticmethod
    def draw_boxes_on_image(image, bboxes, classes, scores, img_meta):
        if classes is None:
            classes = [None] * len(bboxes)
        # 用于保存叠加了box的图像
        image_with_boxes = image.copy()
        for (x0, y0, x1, y1), category, score in zip(bboxes, classes, scores):
            img_h, img_w = image_with_boxes.shape[:2]
            x0 = max(int(x0), 0)
            y0 = max(int(y0), 0)
            x1 = min(int(x1), img_w)
            y1 = min(int(y1), img_h)
            # 画边框
            cv2.rectangle(image_with_boxes, (x0, y0), (x1, y1), (0, 255, 0), 2)
            # 添加caption和score
            if category is not None:
                if not isinstance(category, str):
                    category = img_meta['class_dict'][category] if 'class_dict' in img_meta else str(category)
                cv2.putText(image_with_boxes, category, (x0, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(image_with_boxes, '{:.3f}'.format(score), (x0, y0 + 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return image_with_boxes
