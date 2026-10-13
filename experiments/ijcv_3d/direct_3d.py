"""Direct 3D lane decoding control for the journal model.

The image backbone, image encoder and lane-query decoder deliberately keep the
same module names and shapes as :class:`CondLSTR3D`. Their weights can
therefore be copied from a baseline checkpoint to hold the representation O
fixed while only this coordinate decoder is fitted.
"""
import copy

import numpy as np
import torch
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modeling.models.backbones.stdcnet import STDCNet
from modeling.models.models.lane.utils import PositionEmbeddingProj
from experiments.ijcv_3d.model import AttentionBlock


def interpolate_lane(lane, y_anchors):
    """Interpolate one ego-frame lane at fixed y anchors.

    Returns x/z coordinates and a visibility mask.  Duplicate y values are
    averaged, which avoids a division by zero without silently discarding a
    valid lane.
    """
    lane = torch.as_tensor(lane, device=y_anchors.device, dtype=torch.float32)
    empty = torch.zeros((len(y_anchors), 2), device=y_anchors.device)
    invisible = torch.zeros(len(y_anchors), device=y_anchors.device, dtype=torch.bool)
    if lane.ndim != 2 or lane.shape[-1] < 3:
        return empty, invisible
    lane = lane[:, :3]
    lane = lane[torch.isfinite(lane).all(-1)]
    if len(lane) < 2:
        return empty, invisible
    order = lane[:, 1].argsort(stable=True)
    lane = lane[order]
    unique_y, inverse, counts = torch.unique_consecutive(
        lane[:, 1], return_inverse=True, return_counts=True)
    sums = lane.new_zeros((len(unique_y), 2))
    sums.index_add_(0, inverse, lane[:, (0, 2)])
    xz = sums / counts[:, None]
    if len(unique_y) < 2:
        return empty, invisible
    visible = (y_anchors >= unique_y[0]) & (y_anchors <= unique_y[-1])
    right = torch.searchsorted(unique_y, y_anchors).clamp(1, len(unique_y) - 1)
    left = right - 1
    scale = (y_anchors - unique_y[left]) / (unique_y[right] - unique_y[left]).clamp_min(1e-6)
    coords = xz[left] + scale[:, None] * (xz[right] - xz[left])
    coords = torch.where(visible[:, None], coords, torch.zeros_like(coords))
    return coords, visible


class Direct3DMatcher(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    @torch.no_grad()
    def cost_matrix(self, obj, cls, coords, vis, target):
        labels = target['labels'].long()
        if not len(labels):
            return obj.new_zeros((len(obj), 0), dtype=torch.float32)
        ignored = labels == 255
        safe_labels = labels.masked_fill(ignored, 0)
        cost = self.weights['obj'] * -obj.float().log_softmax(-1)[:, 0, None]
        cost = cost + self.weights['cls'] * (
            -cls.float().log_softmax(-1)[:, safe_labels] * (~ignored)[None])
        q, m, p = len(obj), len(labels), coords.shape[-2]
        pred_vis = vis.float()[:, None].expand(q, m, p)
        gt_vis = target['visible'].float()[None].expand(q, m, p)
        vis_cost = F.binary_cross_entropy_with_logits(pred_vis, gt_vis, reduction='none').mean(-1)
        valid = target['visible'].float()[None, :, :, None]
        regression = (coords.float()[:, None] - target['coords'].float()[None]).abs()
        regression = (regression * valid).sum((-1, -2)) / (2 * valid.sum((-1, -2)).clamp_min(1))
        return cost + self.weights['visibility'] * vis_cost + self.weights['coordinates'] * regression

    @torch.no_grad()
    def forward(self, outputs, targets):
        matches = []
        for i, target in enumerate(targets):
            cost = self.cost_matrix(outputs['object'][i], outputs['category'][i],
                                    outputs['coordinates'][i], outputs['visibility'][i], target)
            if not torch.isfinite(cost).all():
                raise FloatingPointError('Non-finite direct-3D Hungarian matching cost')
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
            matches.append((torch.as_tensor(rows, device=cost.device),
                            torch.as_tensor(cols, device=cost.device)))
        return matches


class Direct3DLoss(nn.Module):
    def __init__(self, points, y_min, y_max, eos_coef=0.4):
        super().__init__()
        self.register_buffer('y_anchors', torch.linspace(y_min, y_max, points), persistent=False)
        self.weights = dict(obj=5., cls=1., visibility=1., coordinates=1.)
        self.matcher = Direct3DMatcher(self.weights)
        self.eos_coef = eos_coef

    def targets(self, lane_points, lane_attris):
        result = []
        for lanes, labels in zip(lane_points, lane_attris):
            sampled = [interpolate_lane(lane, self.y_anchors) for lane in lanes]
            keep = [i for i, (_, visible) in enumerate(sampled) if visible.sum() >= 2]
            if keep:
                coords = torch.stack([sampled[i][0] for i in keep])
                visible = torch.stack([sampled[i][1] for i in keep])
                labels = torch.as_tensor(labels, device=self.y_anchors.device, dtype=torch.long)[keep]
            else:
                coords = self.y_anchors.new_zeros((0, len(self.y_anchors), 2))
                visible = torch.zeros((0, len(self.y_anchors)), device=self.y_anchors.device, dtype=torch.bool)
                labels = torch.empty(0, device=self.y_anchors.device, dtype=torch.long)
            result.append(dict(coords=coords, visible=visible, labels=labels))
        return result

    def forward(self, outputs, lane_points, lane_attris, img_metas=None):
        targets = self.targets(lane_points, lane_attris)
        indices = self.matcher(outputs, targets)
        batch, queries = outputs['object'].shape[:2]
        object_target = torch.ones((batch, queries), device=outputs['object'].device, dtype=torch.long)
        for i, (source, _) in enumerate(indices):
            object_target[i, source] = 0
        obj_weight = outputs['object'].new_tensor([1., self.eos_coef])
        losses = {'direct_object': self.weights['obj'] * F.cross_entropy(
            outputs['object'].float().transpose(1, 2), object_target, weight=obj_weight)}
        normalizer = outputs['object'].new_tensor(float(sum(len(target['labels']) for target in targets)))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(normalizer)
            normalizer = normalizer / dist.get_world_size()
        normalizer = normalizer.clamp_min(1)
        class_weight = None if not img_metas else img_metas[0].get('class_weight')
        if class_weight is not None:
            class_weight = class_weight.to(device=outputs['object'].device, dtype=torch.float32)
        category = outputs['category'].sum() * 0
        visibility = outputs['visibility'].sum() * 0
        coordinates = outputs['coordinates'].sum() * 0
        for i, ((source, target_ids), target) in enumerate(zip(indices, targets)):
            if not len(source):
                continue
            labels = target['labels'][target_ids]
            valid_labels = labels != 255
            if valid_labels.any():
                category = category + F.cross_entropy(
                    outputs['category'][i, source[valid_labels]].float(), labels[valid_labels],
                    weight=class_weight, reduction='sum')
            target_vis = target['visible'][target_ids]
            visibility = visibility + F.binary_cross_entropy_with_logits(
                outputs['visibility'][i, source].float(), target_vis.float(), reduction='sum') / len(self.y_anchors)
            valid = target_vis[..., None].float()
            error = ((outputs['coordinates'][i, source].float() - target['coords'][target_ids]).abs() * valid)
            coordinates = coordinates + (error.sum((-1, -2)) /
                                             (2 * valid.sum((-1, -2)).clamp_min(1))).sum()
        losses['direct_category'] = self.weights['cls'] * category / normalizer
        losses['direct_visibility'] = self.weights['visibility'] * visibility / normalizer
        losses['direct_coordinates'] = self.weights['coordinates'] * coordinates / normalizer
        return losses


class Direct3DHead(nn.Module):
    def __init__(self, channels, queries, points, x_scale, z_scale, num_classes=21):
        super().__init__()
        self.queries, self.points = queries, points
        self.x_scale, self.z_scale = x_scale, z_scale
        self.object = nn.Linear(channels, 2)
        self.category = nn.Linear(channels, num_classes)
        self.coordinates = nn.Linear(channels, points * 2)
        self.visibility = nn.Linear(channels, points)

    def forward(self, tokens):
        coords = self.coordinates(tokens).reshape(len(tokens), self.queries, self.points, 2)
        scale = coords.new_tensor([self.x_scale, self.z_scale])
        return dict(object=self.object(tokens), category=self.category(tokens),
                    coordinates=coords.tanh() * scale, visibility=self.visibility(tokens))


class LaneDirect3D(nn.Module):
    common_prefixes = ('img_backbone.', 'encoder.', 'lane_decoder.', 'lane_queries',
                       'bev_queries', 'position.')

    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        c = config['channels']
        assert c == 256, 'Inherited STDCNet feature output has 256 channels'
        self.img_backbone = STDCNet('ResNet34', pretrained=False)
        block = dict(channels=c, heads=config['heads'], ff=config['ff_channels'], dropout=config['dropout'])
        self.encoder = nn.ModuleList(AttentionBlock(**block) for _ in range(config['encoder_layers']))
        self.lane_decoder = nn.ModuleList(AttentionBlock(cross=True, **block)
                                          for _ in range(config['lane_decoder_layers']))
        self.lane_queries = nn.Parameter(torch.randn(1, config['queries'], c))
        h, w = config['bev_feature_shape']
        self.bev_queries = nn.Parameter(torch.randn(1, c, h, w))
        self.position = PositionEmbeddingProj(embed_dim=c, x_bound=[-10, 10, 20 / w],
                                             y_bound=[3, 103, 100 / h],
                                             z_bound=[-5, 5, 10 / config['height_bins']],
                                             d_bound=[0, 100, 100 / config['depth_bins']], LID=True)
        self.direct_decoder = nn.ModuleList(AttentionBlock(**block)
                                            for _ in range(config['direct_decoder_layers']))
        self.head = Direct3DHead(c, config['queries'], config['direct_points'],
                                 config['direct_x_scale'], config['direct_z_scale'])
        self.loss = Direct3DLoss(config['direct_points'], config['direct_y_min'],
                                 config['direct_y_max'], config['eos_coef'])
        self._common_frozen = False

    def block(self, layer, *args):
        if self.training and self.config['checkpoint_attention']:
            return checkpoint(layer, *args, use_reentrant=False)
        return layer(*args)

    def forward(self, batch):
        img, metas = batch['img'], batch['img_metas']
        for meta in metas:
            meta['input_shape'] = img.shape[-2:]
        image = self.img_backbone(img)
        b = len(image)
        bev = self.bev_queries.expand(b, -1, -1, -1)
        _, image_pos = self.position(bev, image, batch['depth_prior'], metas)
        flatten = lambda tensor: tensor.flatten(2).transpose(1, 2)
        memory, image_pos = flatten(image), flatten(image_pos)
        for layer in self.encoder:
            memory = self.block(layer, memory, image_pos)
        query_pos = self.lane_queries.expand(b, -1, -1)
        lanes = torch.zeros_like(query_pos)
        for layer in self.lane_decoder:
            lanes = self.block(layer, lanes, query_pos, memory, image_pos)
        direct = lanes
        for layer in self.direct_decoder:
            direct = self.block(layer, direct, query_pos)
        outputs = self.head(direct)
        if self.training:
            return self.loss(outputs, batch['lane_points'], batch['lane_attris'], metas)
        return self.postprocess(outputs)

    def postprocess(self, outputs):
        object_score = outputs['object'].float().softmax(-1)[..., 0]
        class_score, category = outputs['category'].float().softmax(-1).max(-1)
        score = (object_score * class_score).sqrt()
        result_points, result_attris, result_scores = [], [], []
        anchors = self.loss.y_anchors.detach().cpu().numpy()
        for sample in range(len(object_score)):
            points, attris, scores = [], [], []
            for query in torch.where(object_score[sample] >= self.config['score_threshold'])[0]:
                visible = outputs['visibility'][sample, query].float().sigmoid() >= self.config['direct_visibility_threshold']
                if visible.sum() < 2:
                    continue
                xz = outputs['coordinates'][sample, query, visible].float().detach().cpu().numpy()
                xyz = np.column_stack((xz[:, 0], anchors[visible.detach().cpu().numpy()], xz[:, 1]))
                points.append(xyz)
                attris.append(int(category[sample, query]))
                scores.append(float(score[sample, query]))
            result_points.append(points); result_attris.append(attris); result_scores.append(scores)
        return dict(lane_points=result_points, lane_attris=result_attris,
                    lane_scores=result_scores, lane_points_frame='ego')

    def load_common_state(self, state):
        """Copy the modules that produce O from a baseline state dictionary."""
        state = {(key.removeprefix('module.')): value for key, value in state.items()}
        own = self.state_dict()
        selected = {key: value for key, value in state.items()
                    if key.startswith(self.common_prefixes) and key in own}
        expected = {key for key in own if key.startswith(self.common_prefixes)}
        missing = expected - selected.keys()
        shape_errors = {key for key, value in selected.items() if value.shape != own[key].shape}
        if missing or shape_errors:
            raise ValueError(f'Incompatible baseline O state: missing={sorted(missing)}, shape={sorted(shape_errors)}')
        self.load_state_dict(selected, strict=False)
        return sorted(selected)

    def freeze_common(self):
        for name, parameter in self.named_parameters():
            if name.startswith(self.common_prefixes):
                parameter.requires_grad_(False)
        self._common_frozen = True
        self._set_common_eval()

    def _set_common_eval(self):
        self.img_backbone.eval()
        self.encoder.eval()
        self.lane_decoder.eval()
        self.position.eval()

    def train(self, mode=True):
        super().train(mode)
        # Fix BatchNorm buffers and disable dropout in the O-producing path as
        # well as freezing its parameters.  The direct decoder/head still train.
        if mode and self._common_frozen:
            self._set_common_eval()
        return self

    def param_groups(self, lr, weight_decay):
        backbone = [parameter for parameter in self.img_backbone.parameters() if parameter.requires_grad]
        ids = {id(parameter) for parameter in backbone}
        rest = [parameter for parameter in self.parameters() if parameter.requires_grad and id(parameter) not in ids]
        groups = []
        if backbone:
            groups.append(dict(params=backbone, lr=lr * .1, weight_decay=weight_decay))
        if rest:
            groups.append(dict(params=rest, lr=lr, weight_decay=weight_decay))
        return groups
