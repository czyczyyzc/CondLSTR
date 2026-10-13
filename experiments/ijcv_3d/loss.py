"""Focal + Dice lane heat-map loss and paired Hungarian assignment.

Assignment is computed in query chunks to avoid an L*M*H*W allocation.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import sigmoid_focal_loss

from modeling.models.detectors.lane.cond_lstr_vote_3d.loss import (
    CondLSTRVote3DLoss, SetCriterion,
)


def heat_loss(logits, target, alpha=0.25, gamma=2.0):
    """One loss per leading instance, including well-defined empty batches."""
    logits, target = logits.float(), target.float()
    focal = sigmoid_focal_loss(logits, target, alpha=alpha, gamma=gamma)
    focal = focal.flatten(-2).mean(-1)
    pred = logits.sigmoid()
    numerator = 2 * (pred * target).flatten(-2).sum(-1) + 1
    denominator = (pred.square() + target.square()).flatten(-2).sum(-1) + 1
    return focal + 1 - numerator / denominator


class CondLSTR3DMatcher(torch.nn.Module):
    def __init__(self, weights, with_2d=True, query_chunk=4, alpha=0.25, gamma=2.0):
        super().__init__()
        self.weights = weights
        self.with_2d = with_2d
        self.query_chunk = query_chunk
        self.alpha, self.gamma = alpha, gamma

    @torch.no_grad()
    def cost_matrix(self, obj, cls, bev, bev_reg, img, img_reg, target):
        labels = target['gt_lbl_cls'].long()
        ignored = labels == 255
        safe_labels = labels.masked_fill(ignored, 0)
        cost = self.weights['obj_weight'] * -obj.float().log_softmax(-1)[:, target['gt_lbl_obj'].long()]
        cost += self.weights['cls_weight'] * -cls.float().log_softmax(-1)[:, safe_labels] * (~ignored)
        if not len(labels):
            return cost
        views = [(bev, bev_reg, 'bev', 3)]
        if self.with_2d:
            views.append((img, img_reg, 'img', 2))
        for masks, regs, view, channels in views:
            truth = target['gt_msk_' + view].float()
            truth_reg = target['gt_reg_' + view][:, :channels].float()
            for start in range(0, len(obj), self.query_chunk):
                stop = start + self.query_chunk
                pred, tgt = torch.broadcast_tensors(masks[start:stop, None].float(), truth[None])
                heat = heat_loss(pred, tgt, self.alpha, self.gamma)
                offset = (regs[start:stop, None, :channels].float() - truth_reg[None]).abs().sum(2)
                offset = (offset * truth[None]).sum((-1, -2)) / truth.sum((-1, -2)).clamp_min(1)
                cost[start:stop] += self.weights['msk_weight'] * heat + self.weights['reg_weight'] * offset
        return cost

    @torch.no_grad()
    def forward(self, obj, cls, bev, bev_reg, img, img_reg, targets):
        result = []
        for i, target in enumerate(targets):
            cost = self.cost_matrix(obj[i], cls[i], bev[i], bev_reg[i],
                                    img[i] if img is not None else None,
                                    img_reg[i] if img_reg is not None else None, target)
            if not torch.isfinite(cost).all():
                raise FloatingPointError('Non-finite Hungarian matching cost')
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
            result.append((torch.as_tensor(rows, device=obj.device), torch.as_tensor(cols, device=obj.device)))
        return result


class CondLSTR3DCriterion(SetCriterion):
    def __init__(self, *args, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha, self.gamma = alpha, gamma

    def loss_msk(self, msks_bev, msks_img, targets, indices, num_lanes):
        idx = self._get_src_permutation_idx(indices)
        loss = msks_bev.sum() * 0
        views = [(msks_bev, 'bev')]
        if self.with_2d:
            views.append((msks_img, 'img'))
        for masks, view in views:
            truth = torch.cat([target['gt_msk_' + view][j] for target, (_, j) in zip(targets, indices)])
            loss = loss + heat_loss(masks[idx], truth, self.alpha, self.gamma).sum() / num_lanes
        return loss


class CondLSTR3DLoss(CondLSTRVote3DLoss):
    def __init__(self, image_supervision=True, paired_matching=True, line_width=16,
                 alpha=0.25, gamma=2.0, eos_coef=0.4):
        weights = dict(obj_weight=5., cls_weight=1., msk_weight=1., reg_weight=1.)
        # Keep targets and filtering identical when toggling image supervision.
        super().__init__(min_points=2, line_width=line_width, weight_dict=weights,
                         eos_coef=eos_coef, with_2d=True)
        matcher = CondLSTR3DMatcher(weights, with_2d=paired_matching, alpha=alpha, gamma=gamma)
        self.criterion = CondLSTR3DCriterion(matcher, **weights, eos_coef=eos_coef,
                                                line_width=line_width, with_2d=image_supervision,
                                                alpha=alpha, gamma=gamma)
