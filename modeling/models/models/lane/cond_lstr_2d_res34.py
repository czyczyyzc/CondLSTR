import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.models import backbones, detectors


class CondLSTR2DRes34(nn.Module):

    def __init__(self, num_classes=21, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CondLSTR2DRes34, self).__init__()
        norm_cfg = dict(type='SyncBN', requires_grad=True) if norm_layer == nn.SyncBatchNorm \
            else dict(type='BN', requires_grad=True)

        img_backbone = dict(
            name='STDCNet',
            backbone='ResNet34',
            pretrained=False,
            norm_layer=norm_layer,
            use_checkpoint=False,
        )
        det_backbone = dict(
            name='Transformer',
            in_channels=256,
            src_shape=(24, 42),
            tgt_shape=(20, 1),
            d_model=256,
            n_heads=8,
            num_encoder_layers=2,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            normalize_before=False,
            return_intermediate_dec=True,
            src_pos_encode='sine',
            tgt_pos_encode='learned',
            use_fix_encode=False,
            use_checkpoint=True,
            src_down_scale=None,
            mem_down_scale=None,
        )
        detector = dict(
            name='CondLSTR2D',
            in_channels=(256, 256),
            num_classes=num_classes,
            head_layers=1,
            disable_coords=True,
            branch_channels=256,
            min_points=2,
            line_width=16,
            score_thresh=0.7,
            eos_coef=0.4,
            mask_downscale=1,
            with_smooth=False
        )
        self.img_backbone = backbones.create(**img_backbone)
        self.det_backbone = backbones.create(**det_backbone)
        self.detector = detectors.create(**detector)
        self.num_classes = num_classes
        self.init_weights()

    def init_weights(self):
        return

    def get_param_groups(self, lr, weight_decay=None):
        exclude = lambda n, p: p.ndim < 2 or 'bn' in n or 'ln' in n or 'norm' in n or 'bias' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = [(n, p) for n, p in self.named_parameters() if 'img_backbone' not in n and p.requires_grad]
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        param_groups = [
            {'params': gain_or_bias_params, 'lr': lr, 'weight_decay': 0.},
            {'params': rest_params, 'lr': lr, 'weight_decay': weight_decay},
        ]
        param_groups.extend(self.img_backbone.get_param_groups(lr, weight_decay))
        return param_groups

    def forward(self, data_dict):
        img, img_metas = data_dict['img'], data_dict['img_metas']
        img_mask = data_dict['img_mask'] if self.training else None
        lane_attris = data_dict.get('lane_attris', None) if self.training else None

        B, C, H, W = img.size()
        pad_mask = torch.ones((B, H, W), dtype=torch.bool, device=img.device)
        for i, img_meta in enumerate(img_metas):
            h, w = img_meta['img_shape'][:2]
            pad_mask[i, :h, :w] = False

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        img_feat = self.img_backbone(img)
        src_mask = F.interpolate(pad_mask[:, None].float(), size=img_feat.shape[-2:], mode='nearest')[:, 0].bool()

        enc_outs = self.det_backbone.forward_encoder(img_feat, src_mask)
        det_feat = self.det_backbone.forward_decoder(enc_outs)[0]

        enc_feat = enc_outs[0].view(*img_feat[-1].shape[-2:], B, -1).permute(2, 3, 0, 1).contiguous()
        det_feat = [x.squeeze(-1).transpose(1, 2) for x in det_feat]

        f_mask = [enc_feat] * len(det_feat)
        f_hm = det_feat

        ret_dict = self.detector(f_mask, f_hm, img_metas, img_mask, lane_attris)
        return ret_dict


# det_feat, _, _, _, attn = self.det_backbone.forward_decoder(enc_outs)
#
# import os
# import os.path as osp
# import cv2
# import numpy as np
# for i, (attn_img, img_meta) in enumerate(zip(attn, img_metas)):
#     im = img[i].permute(1, 2, 0) * torch.tensor([58.395, 57.12, 57.375], device=img.device) + torch.tensor([123.675, 116.28, 103.53], device=img.device)
#     im = im.byte().cpu().numpy()[:, :, [2, 1, 0]]
#     # mm = img_mask[i, 0].cpu().numpy()
#     # im[(mm >= 1) & (mm < 255)] = np.array([255, 0, 0])
#
#     img_name = img_meta['filename']
#     img_name = osp.splitext(osp.basename(img_name))[0]
#     img_path = os.path.join('/data/gpfs/projects/punim1962/attn_map', img_name)
#     os.makedirs(img_path, exist_ok=True)
#     for j, attn_q in enumerate(attn_img):
#         attn_q = attn_q.reshape(60, 120).cpu().numpy()
#         attn_q = (attn_q - attn_q.min()) / (attn_q.max() - attn_q.min())
#         attn_q = (attn_q * 255).astype(np.uint8)
#         cv2.imwrite(os.path.join(img_path, 'query_{}.png'.format(j)), attn_q)
#     cv2.imwrite(os.path.join(img_path, '{}.jpg'.format(img_name)), im)


# print('###############################################################33')
# print(img_metas)
# print(img.shape)
# print(img_mask.shape)
# import os
# import shutil
# import numpy as np
# from PIL import Image
# shutil.rmtree('/data/sets/img_mask')
# os.makedirs('/data/sets/img_mask', exist_ok=True)
# for i in range(len(img_mask)):
#     im = img[i].permute(1, 2, 0) * torch.tensor([58.395, 57.12, 57.375], device=img.device) + torch.tensor([123.675, 116.28, 103.53], device=img.device)
#     im = im.byte().cpu().numpy()
#     mm = img_mask[i, 0].cpu().numpy()
#     im[mm == 255] = np.array([0, 0, 255])
#     im[(mm >= 1) & (mm < 255)] = np.array([255, 0, 0])
#     h, w = img_metas[i]['img_shape'][:2]
#     im[h:, w:] = np.array([0, 0, 255])
#     image = Image.fromarray(im, mode="RGB")
#     image.save(os.path.join('/data/sets/img_mask', 'img_{}.png'.format(i)))
# import sys
# sys.exit()
