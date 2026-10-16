import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.models import backbones, detectors
from .utils import PositionEmbedding, PositionEmbeddingProj


class CondLSTR3DRes34No2D(nn.Module):

    def __init__(self, num_classes=5, use_checkpoint=False, distributed=False, **kwargs):
        super(CondLSTR3DRes34No2D, self).__init__()
        norm_layer = nn.SyncBatchNorm if distributed else nn.BatchNorm2d

        img_backbone = dict(
            name='STDCNet',
            backbone='ResNet34',
            pretrained=False,
            norm_layer=norm_layer,
            use_checkpoint=False,
        )
        det_backbone = dict(
            name='SwapTransformer',
            in_channels=256,
            src_shape=None,
            tgt_shape=None,
            d_model=256,
            n_heads=8,
            num_memory_embeds=80,
            num_encoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            normalize_before=False,
            return_intermediate_dec=True,
            src_pos_encode='none',
            tgt_pos_encode='none',
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
            line_width=16,
            score_thresh=0.7,
            eos_coef=0.4,
            mask_downscale=1,
            with_smooth=False,
            with_2d=False
        )
        self.img_backbone = backbones.create(**img_backbone)
        self.det_backbone = backbones.create(**det_backbone)
        self.detector = detectors.create(**detector)
        self.num_classes = num_classes

        self.pos_embedding = PositionEmbeddingProj(
            embed_dim=256,
            x_bound=[-10, 10, 0.625],
            y_bound=[3, 103, 2.0],
            z_bound=[-5, 5, 0.2],
            d_bound=[0, 100, 2.0],
            LID=True,
        )

        # self.pos_embedding = PositionEmbedding(
        #     num_pos_feats=256,
        #     ego_range=[-20, 0, -10, 20, 200, 10],
        # )

        self.bev_feature = nn.Parameter(torch.randn((1, 256, 50, 32)))
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
        img_feat, bev_feat, mem_feat = self.det_backbone(
            src=img_feat, tgt=bev_feat, src_mask=pad_mask, src_pos_embed=img_embd, tgt_pos_embed=bev_embd)
        det_feat = [x[:80].transpose(0, 1).contiguous() for x in mem_feat]

        ret_dict = self.detector(bev_feat, img_feat, det_feat, img_metas, bev_mask, img_mask, lane_attris)
        return ret_dict
