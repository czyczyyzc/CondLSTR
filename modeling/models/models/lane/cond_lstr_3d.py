"""Unified CondLSTR journal model.

The image-view and BEV branches are trained together.  Both branches use the
same image-conditioned lane representations, and their targets participate in
one paired Hungarian assignment.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from modeling.models.backbones.stdcnet import STDCNet
from modeling.models.models.lane.utils import PositionEmbeddingProj
from modeling.models.detectors.lane.cond_lstr_vote_3d import CondLSTRVote3D
from modeling.models.detectors.lane.cond_lstr_vote_3d.journal_loss import CondLSTR3DLoss


class AttentionBlock(nn.Module):
    def __init__(self, cross=False, channels=256, heads=8, ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True) if cross else None
        self.norms = nn.ModuleList(nn.LayerNorm(channels) for _ in range(3 if cross else 2))
        self.ff = nn.Sequential(nn.Linear(channels, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, channels))
        self.drop = nn.Dropout(dropout)

    def forward(self, value, pos, memory=None, memory_pos=None):
        q = value + pos
        value = self.norms[0](value + self.drop(self.self_attn(q, q, value, need_weights=False)[0]))
        if self.cross_attn is not None:
            key = memory if memory_pos is None else memory + memory_pos
            value = self.norms[1](value + self.drop(self.cross_attn(value + pos, key, memory, need_weights=False)[0]))
        return self.norms[-1](value + self.drop(self.ff(value)))


class CondLSTR3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        c = config['channels']
        assert c == 256, 'Inherited STDCNet feature output has 256 channels'
        self.img_backbone = STDCNet('ResNet34', pretrained=False)
        block = dict(channels=c, heads=config['heads'], ff=config['ff_channels'], dropout=config['dropout'])
        self.encoder = nn.ModuleList(AttentionBlock(**block) for _ in range(config['encoder_layers']))
        self.lane_decoder = nn.ModuleList(AttentionBlock(cross=True, **block) for _ in range(config['lane_decoder_layers']))
        self.bev_decoder = nn.ModuleList(AttentionBlock(cross=True, **block) for _ in range(config['bev_decoder_layers']))
        self.lane_queries = nn.Parameter(torch.randn(1, config['queries'], c))
        h, w = config['bev_feature_shape']
        self.bev_queries = nn.Parameter(torch.randn(1, c, h, w))
        self.position = PositionEmbeddingProj(embed_dim=c, x_bound=[-10, 10, 20 / w],
                                             y_bound=[3, 103, 100 / h],
                                             z_bound=[-5, 5, 10 / config['height_bins']],
                                             d_bound=[0, 100, 100 / config['depth_bins']], LID=True)
        loss = CondLSTR3DLoss(
            image_supervision=config['image_supervision'],
            paired_matching=config['paired_matching'],
            line_width=config['line_width'],
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma'],
            eos_coef=config['eos_coef'],
        )
        self.detector = CondLSTRVote3D(
            in_channels=(c, c),
            num_classes=21,
            head_layers=1,
            disable_coords=True,
            branch_channels=c,
            min_points=2,
            line_width=config['line_width'],
            score_thresh=config['score_threshold'],
            eos_coef=config['eos_coef'],
            mask_downscale=1,
            with_smooth=False,
            with_2d=True,
            loss_module=loss,
        )

    def block(self, layer, *args):
        if self.training and self.config['checkpoint_attention']:
            return checkpoint(layer, *args, use_reentrant=False)
        return layer(*args)

    def forward(self, batch):
        img, metas = batch['img'], batch['img_metas']
        for meta in metas:
            meta['input_shape'] = img.shape[-2:]
        image = self.img_backbone(img)
        b, c, ih, iw = image.shape
        bev = self.bev_queries.expand(b, -1, -1, -1)
        bp, ip = self.position(bev, image, batch['depth_prior'], metas)
        flatten = lambda t: t.flatten(2).transpose(1, 2)
        memory, ip = flatten(image), flatten(ip)
        for layer in self.encoder:
            memory = self.block(layer, memory, ip)
        qp = self.lane_queries.expand(b, -1, -1)
        lanes = torch.zeros_like(qp)
        for layer in self.lane_decoder:
            lanes = self.block(layer, lanes, qp, memory, ip)
        bev, bp = flatten(bev), flatten(bp)
        transform_memory = self.config.get('transform_memory', 'lane_representations')
        if transform_memory == 'lane_representations':
            source, source_pos = lanes, None
            detector_tokens = lanes
        elif transform_memory == 'generic_tokens':
            # Reuse the learned query embeddings as image-independent memory.
            # The lane decoder is still executed so this control has the same
            # inference graph and attention cost as the complete model.
            source, source_pos = qp, None
            detector_tokens = qp
        elif transform_memory == 'dense_image':
            source, source_pos = memory, ip
            detector_tokens = lanes
        else:
            raise ValueError(f'Unknown transform_memory={transform_memory!r}')
        for layer in self.bev_decoder:
            bev = self.block(layer, bev, bp, source, source_pos)
        bh, bw = self.config['bev_feature_shape']
        bev = bev.transpose(1, 2).reshape(b, c, bh, bw)
        memory = memory.transpose(1, 2).reshape(b, c, ih, iw)
        if not self.training:
            raw = self.detector.forward_raw(bev, memory, detector_tokens, metas[0]['bev_shape'], img.shape[-2:])
            # Voting accumulates many pixels and exports NumPy arrays; keep it FP32.
            with torch.autocast(img.device.type, enabled=False):
                points, attris, scores = self.detector.postprocess(*[x.float() for x in raw], metas)
            return dict(lane_points=points, lane_attris=attris, lane_scores=scores)
        # Only the final decoder output is supervised.
        return self.detector(bev, memory, detector_tokens, metas,
                             batch.get('bev_mask') if self.training else None,
                             batch.get('img_mask') if self.training else None,
                             batch.get('lane_attris') if self.training else None)

    def param_groups(self, lr, weight_decay):
        backbone = list(self.img_backbone.parameters())
        ids = {id(p) for p in backbone}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [dict(params=backbone, lr=lr * .1, weight_decay=weight_decay),
                dict(params=rest, lr=lr, weight_decay=weight_decay)]


def build_model(config):
    variant = config.get('model_variant', 'complete')
    if variant == 'complete':
        return CondLSTR3D(config)
    if variant == 'direct_3d':
        from .direct_3d import LaneDirect3D
        return LaneDirect3D(config)
    raise ValueError(f'Unknown IJCV 3D model_variant={variant!r}')
