import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.models import backbones, detectors
from .utils import PositionEmbedding, PositionEmbeddingProj


class CondLSTR3DRes34TR(nn.Module):

    def __init__(self, num_classes=5, use_checkpoint=False, distributed=False, **kwargs):
        super(CondLSTR3DRes34TR, self).__init__()
        norm_layer = nn.SyncBatchNorm if distributed else nn.BatchNorm2d

        img_backbone = dict(
            name='STDCNet',
            backbone='ResNet34',
            pretrained=False,
            norm_layer=norm_layer,
            use_checkpoint=False,
        )
        bev_backbone = dict(
            name='Transformer',
            in_channels=256,
            src_shape=None,
            tgt_shape=None,
            d_model=256,
            n_heads=8,
            num_encoder_layers=0,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            normalize_before=False,
            return_intermediate_dec=True,
            src_pos_encode='none',
            tgt_pos_encode='none',
            src_down_scale=2,
            mem_down_scale=None,
            use_fix_encode=False,
            use_input_proj=False,
            use_checkpoint=use_checkpoint,
        )
        det_backbone = dict(
            name='Transformer',
            in_channels=256,
            src_shape=(24, 42),
            tgt_shape=(20, 1),
            d_model=256,
            n_heads=8,
            num_encoder_layers=0,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            normalize_before=False,
            return_intermediate_dec=True,
            src_pos_encode='sine',
            tgt_pos_encode='learned',
            src_down_scale=None,
            mem_down_scale=None,
            use_fix_encode=False,
            use_input_proj=False,
            use_checkpoint=use_checkpoint,
        )
        # detector = dict(
        #     name='CondLSTR3D',
        #     in_channels=(256, 256),
        #     num_classes=num_classes,
        #     head_layers=1,
        #     disable_coords=True,
        #     branch_channels=256,
        #     min_points=2,
        #     line_width=16,
        #     score_thresh=0.7,
        #     eos_coef=0.4,
        #     mask_downscale=1,
        #     with_smooth=False,
        #     with_2d=True,
        # )
        detector = dict(
            name='CondLSTRVote3D',
            in_channels=(256, 256),
            num_classes=num_classes,
            head_layers=1,
            disable_coords=True,
            branch_channels=256,
            min_points=2,
            line_width=32,
            score_thresh=0.5,
            eos_coef=0.4,
            mask_downscale=1,
            with_smooth=False,
            with_2d=False
        )
        self.img_backbone = backbones.create(**img_backbone)
        self.bev_backbone = backbones.create(**bev_backbone)
        self.det_backbone = backbones.create(**det_backbone)
        self.detector = detectors.create(**detector)
        self.num_classes = num_classes

        self.pos_embedding = PositionEmbeddingProj(
            embed_dim=256,
            x_bound=[-10, 10, 0.3125],
            y_bound=[3, 103, 0.78125],
            z_bound=[-5, 5, 0.1],
            d_bound=[0, 100, 0.78125],
            LID=True,
        )

        # self.pos_embedding = PositionEmbedding(
        #     num_pos_feats=256,
        #     ego_range=[-20, 0, -10, 20, 200, 10],
        # )

        self.bev_feature = nn.Parameter(torch.randn((1, 256, 128, 64)))
        self.init_weights()

    def get_param_groups(self, lr, weight_decay=None):
        exclude = lambda n, p: p.ndim < 2 or 'bn' in n or 'ln' in n or 'norm' in n or 'bias' in n or 'embed' in n
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

    def init_weights(self):
        nn.init.normal_(self.bev_feature)
        return

    def forward(self, data_dict):
        img, img_metas = data_dict['img'], data_dict['img_metas']
        depth_prior = data_dict['depth_prior']
        img_mask = data_dict['img_mask'] if self.training else None
        bev_mask = data_dict['bev_mask'] if self.training else None
        lane_attris = data_dict.get('lane_attris', None) if self.training else None

        B, C, H, W = img.size()
        pad_mask = torch.ones((B, H, W), dtype=torch.bool, device=img.device)
        for i, img_meta in enumerate(img_metas):
            img_meta.update({'input_shape': img.shape[-2:]})
            h, w = img_meta['img_shape'][:2]
            pad_mask[i, :h, :w] = False

        img_feat = self.img_backbone(img)
        bev_feat = self.bev_feature.repeat(B, 1, 1, 1)
        bev_embd, img_embd = self.pos_embedding(bev_feat, img_feat, depth_prior, img_metas)

        pad_mask = F.interpolate(pad_mask[:, None].float(), size=img_feat.shape[-2:], mode='nearest')[:, 0].bool()
        enc_outs = self.bev_backbone.forward_encoder(img_feat, src_mask=pad_mask, src_pos_embed=img_embd)
        bev_feat = self.bev_backbone.forward_decoder(enc_outs, tgt=bev_feat, tgt_pos_embed=bev_embd)[0]

        det_feat = self.det_backbone(bev_feat[-1])[0]
        det_feat = [x.squeeze(-1).transpose(1, 2) for x in det_feat]
        ret_dict = self.detector(bev_feat, img_feat, det_feat, img_metas, bev_mask, img_mask, lane_attris)
        return ret_dict


# class CondLSTR3DRes34(nn.Module):

#     def __init__(self, num_classes=5, use_checkpoint=False, distributed=False, **kwargs):
#         super(CondLSTR3DRes34, self).__init__()
#         norm_layer = nn.SyncBatchNorm if distributed else nn.BatchNorm2d

#         img_backbone = dict(
#             name='STDCNet',
#             backbone='ResNet34',
#             pretrained=False,
#             norm_layer=norm_layer,
#             use_checkpoint=False,
#         )
#         det_backbone = dict(
#             name='SwapTransformer',
#             in_channels=256,
#             src_shape=None,
#             tgt_shape=None,
#             d_model=256,
#             n_heads=8,
#             num_memory_embeds=256,
#             num_encoder_layers=4,
#             dim_feedforward=1024,
#             dropout=0.1,
#             activation='relu',
#             normalize_before=False,
#             return_intermediate_dec=True,
#             src_pos_encode='none',
#             tgt_pos_encode='none',
#             use_fix_encode=False,
#             use_input_proj=False,
#             use_checkpoint=use_checkpoint,
#         )
#         detector = dict(
#             name='CondLSTR3D',
#             in_channels=(256, 256),
#             num_classes=num_classes,
#             head_layers=1,
#             disable_coords=True,
#             branch_channels=256,
#             min_points=2,
#             line_width=16,
#             score_thresh=0.7,
#             eos_coef=0.4,
#             mask_downscale=1,
#             with_smooth=False,
#             with_2d=True,
#         )
#         # detector = dict(
#         #     name='CondLSTRVote3D',
#         #     in_channels=(256, 256),
#         #     num_classes=num_classes,
#         #     head_layers=1,
#         #     disable_coords=True,
#         #     branch_channels=256,
#         #     min_points=2,
#         #     line_width=32,
#         #     score_thresh=0.7,
#         #     eos_coef=0.4,
#         #     mask_downscale=1,
#         #     with_smooth=False,
#         #     with_2d=True,
#         # )
#         self.img_backbone = backbones.create(**img_backbone)
#         self.det_backbone = backbones.create(**det_backbone)
#         self.detector = detectors.create(**detector)
#         self.num_classes = num_classes

#         self.pos_embedding = PositionEmbeddingProj(
#             embed_dim=256,
#             x_bound=[-10, 10, 0.3125],
#             y_bound=[3, 103, 0.78125],
#             z_bound=[-5, 5, 0.1],
#             d_bound=[0, 100, 0.78125],
#             LID=True,
#         )

#         # self.pos_embedding = PositionEmbedding(
#         #     num_pos_feats=256,
#         #     ego_range=[-20, 0, -10, 20, 200, 10],
#         # )

#         self.bev_feature = nn.Parameter(torch.randn((1, 256, 128, 64)))
#         self.init_weights()

#     def get_param_groups(self, lr, weight_decay=None):
#         exclude = lambda n, p: p.ndim < 2 or 'bn' in n or 'ln' in n or 'norm' in n or 'bias' in n or 'embed' in n
#         include = lambda n, p: not exclude(n, p)

#         named_parameters = [(n, p) for n, p in self.named_parameters() if 'img_backbone' not in n and p.requires_grad]
#         gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
#         rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

#         param_groups = [
#             {'params': gain_or_bias_params, 'lr': lr, 'weight_decay': 0.},
#             {'params': rest_params, 'lr': lr, 'weight_decay': weight_decay},
#         ]
#         param_groups.extend(self.img_backbone.get_param_groups(lr, weight_decay))
#         return param_groups

#     def init_weights(self):
#         nn.init.normal_(self.bev_feature)
#         return

#     def forward(self, data_dict):
#         img, img_metas = data_dict['img'], data_dict['img_metas']
#         depth_prior = data_dict['depth_prior']
#         img_mask = data_dict['img_mask'] if self.training else None
#         bev_mask = data_dict['bev_mask'] if self.training else None
#         lane_attris = data_dict.get('lane_attris', None) if self.training else None

#         B, C, H, W = img.size()
#         pad_mask = torch.ones((B, H, W), dtype=torch.bool, device=img.device)
#         for i, img_meta in enumerate(img_metas):
#             img_meta.update({'input_shape': img.shape[-2:]})
#             h, w = img_meta['img_shape'][:2]
#             pad_mask[i, :h, :w] = False

#         img_feat = self.img_backbone(img)
#         bev_feat = self.bev_feature.repeat(B, 1, 1, 1)
#         bev_embd, img_embd = self.pos_embedding(bev_feat, img_feat, depth_prior, img_metas)

#         pad_mask = F.interpolate(pad_mask[:, None].float(), size=img_feat.shape[-2:], mode='nearest')[:, 0].bool()
#         img_feat, bev_feat, mem_feat = self.det_backbone(
#             src=img_feat, tgt=bev_feat, src_mask=pad_mask, src_pos_embed=img_embd, tgt_pos_embed=bev_embd)
#         det_feat = [x[:20].transpose(0, 1).contiguous() for x in mem_feat]
#         ret_dict = self.detector(bev_feat, img_feat, det_feat, img_metas, bev_mask, img_mask, lane_attris)
#         return ret_dict


# print('###############################################################################')
# import os
# import cv2
# import copy
# import shutil
# import numpy as np

# shutil.rmtree('/data/gpfs/projects/punim1962/img_mask', ignore_errors=True)
# os.makedirs('/data/gpfs/projects/punim1962/img_mask', exist_ok=True)
# for i, (img_msk, bev_msk, lane_attr) in enumerate(zip(img_mask, bev_mask, lane_attris)):
#     ins_msk_bev = bev_msk[:, 0]
#     hgt_msk_bev = bev_msk[:, 1]

#     ins_msk_img = img_msk[:, 0]
#     dpt_msk_img = img_msk[:, 1]

#     e2b_tran = img_metas[i]['calibration']['trans_ego2bev']
#     b2e_tran = img_metas[i]['calibration']['trans_bev2ego']
#     e2i_tran = img_metas[i]['calibration']['trans_ego2img']
#     e2b_tran = torch.from_numpy(e2b_tran).to(img.device).float()
#     b2e_tran = torch.from_numpy(b2e_tran).to(img.device).float()
#     e2i_tran = torch.from_numpy(e2i_tran).to(img.device).float()

#     img_i = img[i].permute(1, 2, 0) * torch.tensor([58.395, 57.12, 57.375], device=img.device) + torch.tensor(
#         [123.675, 116.28, 103.53], device=img.device)
#     img_i = img_i[:, :, [2, 1, 0]].contiguous().byte().cpu().numpy()

#     img_h, img_w = img_i.shape[:2]
#     bev_h, bev_w = bev_msk.shape[-2:]

#     img_i_3d = copy.deepcopy(img_i)
#     map_i_3d = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
#     for j, (ins_msk, hgt_msk) in enumerate(zip(ins_msk_bev, hgt_msk_bev)):
#         ins_pnts = ins_msk.nonzero()
#         map_i_3d[(ins_pnts[:, 0].cpu().numpy(), ins_pnts[:, 1].cpu().numpy())] = np.array([255, 255, 255])

#         ins_hgts = hgt_msk[ins_msk.bool()]
#         ins_pnts = torch.stack([ins_pnts[:, 1], ins_pnts[:, 0], ins_hgts, torch.ones_like(ins_hgts)], dim=0)

#         # print('33333333333333333333333')
#         # print(ins_pnts.t())

#         ins_pnts = torch.matmul(b2e_tran, ins_pnts)
#         ins_pnts = torch.matmul(e2i_tran, ins_pnts)
#         ins_pnts = ins_pnts[:2] / ins_pnts[2:3]
#         ins_pnts = ins_pnts.round().long().cpu().numpy().T
#         ins_pnts = ins_pnts[(ins_pnts[:, 0] >= 0) & (ins_pnts[:, 0] < img_w) &
#                             (ins_pnts[:, 1] >= 0) & (ins_pnts[:, 1] < img_h)]
#         img_i_3d[(ins_pnts[:, 1], ins_pnts[:, 0])] = np.array([0, 0, 255])

#         # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         # print(ins_mask.shape)
#         # print(hgt_mask.shape)
#         # print(ins_clss)
#         # print(ins_pnts)
#         # for k, (p_curr, p_next) in enumerate(zip(ins_pnts[:-1], ins_pnts[1:])):
#         #     p_curr = (int(p_curr[0]), int(p_curr[1]))
#         #     p_next = (int(p_next[0]), int(p_next[1]))
#         #     img_i_0 = cv2.line(img_i_0, p_curr, p_next, color=(255, 0, 0), thickness=4)

#     img_i_2d = copy.deepcopy(img_i)
#     map_i_2d = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
#     for j, (ins_msk, dpt_msk) in enumerate(zip(ins_msk_img, dpt_msk_img)):
#         ins_mask = ins_msk.cpu().numpy()
#         img_i_2d[ins_mask > 0] = np.array([0, 0, 255])

#         ins_pnts = ins_msk.nonzero()
#         ins_dpts = dpt_msk[ins_msk.bool()] + depth_prior[i][ins_msk.bool()]

#         # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         # print(dpt_msk[ins_msk.bool()])

#         # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
#         # print(depth_prior[i][ins_msk.bool()])

#         ins_pnts = torch.stack([ins_pnts[:, 1] * ins_dpts,
#                                 ins_pnts[:, 0] * ins_dpts,
#                                 ins_dpts, torch.ones_like(ins_dpts)], dim=0)

#         # print('4444444444444444')
#         # print(ins_pnts.t())

#         ego2imgs = torch.eye(4, device=img.device)         # (4, 4)
#         ego2imgs[:3] = e2i_tran                            # (3, 4)
#         ins_pnts = torch.inverse(ego2imgs).mm(ins_pnts)    # (4, P)
#         ins_pnts = torch.matmul(e2b_tran, ins_pnts)        # (4, P)

#         # print('5555555555555555')
#         # print(ins_pnts.t())

#         ins_pnts = ins_pnts[:2]
#         ins_pnts = ins_pnts.round().long().cpu().numpy().T
#         ins_pnts = ins_pnts[(ins_pnts[:, 0] >= 0) & (ins_pnts[:, 0] < bev_w) &
#                             (ins_pnts[:, 1] >= 0) & (ins_pnts[:, 1] < bev_h)]
#         map_i_2d[(ins_pnts[:, 1], ins_pnts[:, 0])] = np.array([255, 255, 255])

#     cv2.imwrite('/data/gpfs/projects/punim1962/img_mask/map_{}_3d.jpg'.format(i), map_i_3d)
#     cv2.imwrite('/data/gpfs/projects/punim1962/img_mask/map_{}_2d.jpg'.format(i), map_i_2d)
#     cv2.imwrite('/data/gpfs/projects/punim1962/img_mask/img_{}_3d.jpg'.format(i), img_i_3d)
#     cv2.imwrite('/data/gpfs/projects/punim1962/img_mask/img_{}_2d.jpg'.format(i), img_i_2d)
# import sys
# sys.exit()


# class CondLSTR3DRes34(nn.Module):

#     def __init__(self, num_classes=5, norm_layer=nn.BatchNorm2d, **kwargs):
#         super(CondLSTR3DRes34, self).__init__()
#         norm_cfg = dict(type='SyncBN', requires_grad=True) if norm_layer == nn.SyncBatchNorm \
#             else dict(type='BN', requires_grad=True)

#         img_backbone = dict(
#             name='STDCNet',
#             backbone='ResNet34',
#             pretrained=False,
#             norm_layer=norm_layer,
#             use_checkpoint=False,
#         )
#         det_backbone = dict(
#             name='SwapTransformer',
#             in_channels=256,
#             src_shape=None,
#             tgt_shape=None,
#             d_model=256,
#             n_heads=8,
#             num_memory_embeds=256,
#             num_encoder_layers=4,
#             dim_feedforward=1024,
#             dropout=0.1,
#             activation='relu',
#             normalize_before=False,
#             return_intermediate_dec=True,
#             src_pos_encode='none',
#             tgt_pos_encode='none',
#             use_fix_encode=False,
#             use_checkpoint=False,
#             use_input_proj=False,
#         )
#         detector = dict(
#             name='CondLSTR3D',
#             in_channels=(256, 256),
#             num_classes=num_classes,
#             head_layers=1,
#             disable_coords=True,
#             branch_channels=256,
#             min_points=2,
#             line_width=16,
#             score_thresh=0.7,
#             eos_coef=0.4,
#             mask_downscale=1,
#             with_smooth=False,
#             with_2d=True,
#         )
#         # detector = dict(
#         #     name='CondLSTRVote3D',
#         #     in_channels=(256, 256),
#         #     num_classes=num_classes,
#         #     head_layers=1,
#         #     disable_coords=True,
#         #     branch_channels=256,
#         #     min_points=2,
#         #     line_width=32,
#         #     score_thresh=0.7,
#         #     eos_coef=0.4,
#         #     mask_downscale=1,
#         #     with_smooth=False,
#         #     with_2d=True,
#         # )
#         self.img_backbone = backbones.create(**img_backbone)
#         self.det_backbone = backbones.create(**det_backbone)
#         self.detector = detectors.create(**detector)
#         self.num_classes = num_classes

#         self.d_head = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#             norm_layer(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True),
#         )
#         self.z_head = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#             norm_layer(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True),
#         )
#         self.pos_embedding = PositionEmbedding(
#             num_pos_feats=256,
#             ego_range=[-20, 0, -10, 20, 200, 10],
#         )
#         self.bev_feature = nn.Parameter(torch.randn((1, 256, 128, 64)))
#         self.init_weights()

#     def get_param_groups(self, lr, weight_decay=None):
#         exclude = lambda n, p: p.ndim < 2 or 'bn' in n or 'ln' in n or 'norm' in n or 'bias' in n
#         include = lambda n, p: not exclude(n, p)

#         named_parameters = [(n, p) for n, p in self.named_parameters() if 'img_backbone' not in n and p.requires_grad]
#         gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
#         rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

#         param_groups = [
#             {'params': gain_or_bias_params, 'lr': lr, 'weight_decay': 0.},
#             {'params': rest_params, 'lr': lr, 'weight_decay': weight_decay},
#         ]
#         param_groups.extend(self.img_backbone.get_param_groups(lr, weight_decay))
#         return param_groups

#     def init_weights(self):
#         nn.init.normal_(self.bev_feature)
#         nn.init.normal_(self.d_head[-1].weight, std=0.01)
#         nn.init.normal_(self.z_head[-1].weight, std=0.01)
#         nn.init.constant_(self.d_head[-1].bias, 0)
#         nn.init.constant_(self.z_head[-1].bias, 0)
#         return

#     def forward(self, data_dict):
#         img, img_metas = data_dict['img'], data_dict['img_metas']
#         depth_prior = data_dict['depth_prior']
#         img_mask = data_dict['img_mask'] if self.training else None
#         bev_mask = data_dict['bev_mask'] if self.training else None
#         lane_attris = data_dict.get('lane_attris', None) if self.training else None

#         B, C, H, W = img.size()
#         pad_mask = torch.ones((B, H, W), dtype=torch.bool, device=img.device)
#         for i, img_meta in enumerate(img_metas):
#             img_meta.update({'input_shape': img.shape[-2:]})
#             h, w = img_meta['img_shape'][:2]
#             pad_mask[i, :h, :w] = False

#         img_feat = self.img_backbone(img)
#         bev_feat = self.bev_feature.repeat(B, 1, 1, 1)
#         img_regz = self.d_head(img_feat)
#         bev_regz = self.z_head(bev_feat)

#         bev_embd, img_embd = self.pos_embedding(bev_feat, img_feat, bev_regz, img_regz, depth_prior, img_metas)
#         pad_mask = F.interpolate(pad_mask[:, None].float(), size=img_feat.shape[-2:], mode='nearest')[:, 0].bool()
#         img_feat, bev_feat, mem_feat = self.det_backbone(
#             src=img_feat, tgt=bev_feat, src_mask=pad_mask, src_pos_embed=img_embd, tgt_pos_embed=bev_embd)
#         det_feat = [x[:20].transpose(0, 1).contiguous() for x in mem_feat]
#         ret_dict = self.detector(bev_feat, img_feat, det_feat, img_metas, bev_mask, img_mask, lane_attris)
#         return ret_dict


# class CondLSTR3DRes34(nn.Module):

#     def __init__(self, num_classes=5, norm_layer=nn.BatchNorm2d, **kwargs):
#         super(CondLSTR3DRes34, self).__init__()
#         norm_cfg = dict(type='SyncBN', requires_grad=True) if norm_layer == nn.SyncBatchNorm \
#             else dict(type='BN', requires_grad=True)

#         img_backbone = dict(
#             name='STDCNet',
#             backbone='ResNet34',
#             pretrained=False,
#             norm_layer=norm_layer,
#             use_checkpoint=False,
#         )
#         det_backbone = dict(
#             name='SwapTransformer',
#             in_channels=256,
#             src_shape=None,
#             tgt_shape=(160, 32),
#             d_model=256,
#             n_heads=8,
#             num_memory_embeds=256,
#             num_encoder_layers=4,
#             dim_feedforward=1024,
#             dropout=0.1,
#             activation='relu',
#             normalize_before=False,
#             return_intermediate_dec=True,
#             src_pos_encode='none',
#             tgt_pos_encode='none',
#             use_fix_encode=False,
#             use_checkpoint=False,
#             use_input_proj=False,
#         )
#         detector = dict(
#             name='CondLSTR3D',
#             in_channels=(256, 256),
#             num_classes=num_classes,
#             head_layers=1,
#             disable_coords=True,
#             branch_channels=256,
#             min_points=2,
#             line_width=16,
#             score_thresh=0.7,
#             eos_coef=0.4,
#             mask_downscale=1,
#             with_smooth=False,
#             with_2d=True,
#         )
#         self.img_backbone = backbones.create(**img_backbone)
#         self.det_backbone = backbones.create(**det_backbone)
#         self.detector = detectors.create(**detector)
#         self.num_classes = num_classes

#         self.pos_embedding = PositionEmbedding(embed_dim=256)
#         self.bev_feature = nn.Parameter(torch.randn((1, 256, 200, 40)))
#         nn.init.normal_(self.bev_feature)

#         self.z_head_bev = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#             norm_layer(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True),
#         )
#         self.z_head_img = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#             norm_layer(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True),
#         )
#         self.init_weights()

#     def get_param_groups(self, lr, weight_decay=None):
#         exclude = lambda n, p: p.ndim < 2 or 'bn' in n or 'ln' in n or 'norm' in n or 'bias' in n
#         include = lambda n, p: not exclude(n, p)

#         named_parameters = [(n, p) for n, p in self.named_parameters() if 'img_backbone' not in n and p.requires_grad]
#         gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
#         rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

#         param_groups = [
#             {'params': gain_or_bias_params, 'lr': lr, 'weight_decay': 0.},
#             {'params': rest_params, 'lr': lr, 'weight_decay': weight_decay},
#         ]
#         param_groups.extend(self.img_backbone.get_param_groups(lr, weight_decay))
#         return param_groups

#     def init_weights(self):
#         nn.init.normal_(self.z_head_bev[-1].weight, mean=0, std=0.01)
#         nn.init.normal_(self.z_head_img[-1].weight, mean=0, std=0.01)
#         nn.init.zeros_(self.z_head_bev[-1].bias)
#         nn.init.zeros_(self.z_head_img[-1].bias)

#     def forward(self, data_dict):
#         img, img_metas = data_dict['img'], data_dict['img_metas']
#         depth_prior = data_dict['depth_prior']
#         img_mask = data_dict['img_mask'] if self.training else None
#         bev_mask = data_dict['bev_mask'] if self.training else None
#         lane_attris = data_dict.get('lane_attris', None) if self.training else None

#         B, C, H, W = img.size()
#         pad_mask = torch.ones((B, H, W), dtype=torch.bool, device=img.device)
#         for i, img_meta in enumerate(img_metas):
#             img_meta.update({'input_shape': img.shape[-2:]})
#             h, w = img_meta['img_shape'][:2]
#             pad_mask[i, :h, :w] = False

#         img_feat = self.img_backbone(img)
#         bev_feat = self.bev_feature.repeat(B, 1, 1, 1)
#         bev_regz = self.z_head_bev(bev_feat)
#         img_regz = self.z_head_img(img_feat)
#         bev_size = img_metas[0]['bev_shape'][:2]
#         bev_regz = F.interpolate(bev_regz, size=bev_size, mode='bilinear', align_corners=True)
#         img_regz = F.interpolate(img_regz, size=(H, W), mode='bilinear', align_corners=True)
#         bev_embd, img_embd = self.pos_embedding(bev_feat, img_feat, bev_regz, img_regz, depth_prior, img_metas)

#         pad_mask = F.interpolate(pad_mask[:, None].float(), size=img_feat.shape[-2:], mode='nearest')[:, 0].bool()
#         img_feat, bev_feat, mem_feat = self.det_backbone(
#             src=img_feat, tgt=bev_feat, src_mask=pad_mask, src_pos_embed=img_embd, tgt_pos_embed=bev_embd)
#         det_feat = [x[:20].transpose(0, 1).contiguous() for x in mem_feat]
#         ret_dict = self.detector(bev_feat, img_feat, det_feat, img_metas, bev_mask, img_mask, lane_attris)

#         if self.training:
#             loss_reg_bev = 0.
#             for i, mask in enumerate(bev_mask):
#                 tgt_mask_bev = mask[:, 0].float()                                          # (M, H, W)
#                 tgt_regz_bev = mask[:, 1].float()                                          # (M, H, W)
#                 src_regz_bev = bev_regz[i].repeat(len(mask), 1, 1)                         # (M, H, W)
#                 loss_rgz_bev = F.l1_loss(src_regz_bev, tgt_regz_bev, reduction='none')     # (M, H, W)
#                 loss_rgz_bev = loss_rgz_bev * tgt_mask_bev                                 # (M, H, W)
#                 loss_rgz_bev = loss_rgz_bev.sum(dim=(1, 2)) / tgt_mask_bev.sum(dim=(1, 2)).clamp(min=1.0)  # (M,)
#                 loss_reg_bev = loss_reg_bev + loss_rgz_bev.sum()

#             loss_reg_img = 0.
#             for i, mask in enumerate(img_mask):
#                 tgt_mask_img = mask[:, 0].float()                                          # (M, H, W)
#                 tgt_regz_img = mask[:, 1].float()                                          # (M, H, W)
#                 src_regz_img = img_regz[i].repeat(len(mask), 1, 1)                         # (M, H, W)
#                 loss_rgz_img = F.l1_loss(src_regz_img, tgt_regz_img, reduction='none')     # (M, H, W)
#                 loss_rgz_img = loss_rgz_img * tgt_mask_img                                 # (M, H, W)
#                 loss_rgz_img = loss_rgz_img.sum(dim=(1, 2)) / tgt_mask_img.sum(dim=(1, 2)).clamp(min=1.0)  # (M,)
#                 loss_reg_img = loss_reg_img + loss_rgz_img.sum()

#             loss_reg = (loss_reg_bev + loss_reg_img) / max(sum([len(mask) for mask in bev_mask]), 1.0)
#             ret_dict.update(rgz=loss_reg)
#         return ret_dict
