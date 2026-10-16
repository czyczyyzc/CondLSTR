# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from modeling.models.losses import BinaryDiceLoss
from modeling.models.losses.focal_loss import sigmoid_focal_loss


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, obj_weight: float = 1, cls_weight: float = 1,  msk_weight: float = 1, reg_weight: float = 1,
                 line_width: int = 13, with_2d: bool = True):
        """Creates the matcher
        """
        super().__init__()
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.msk_weight = msk_weight
        self.reg_weight = reg_weight
        self.line_width = line_width
        self.with_2d    = with_2d

        self.dice_loss = BinaryDiceLoss(smooth=1, p=2, reduction='none')

    @torch.no_grad()
    def forward(self, logits_obj, logits_cls, msks_bev, regs_bev, msks_img, regs_img, targets):
        """ Performs the matching
        """
        bs, num_queries = logits_obj.shape[:2]
        device = logits_obj.device

        C_list = []
        for i, tgt in enumerate(targets):
            # We flatten to compute the cost matrices in a batch
            out_prob = logits_obj[i].softmax(-1)                                                                 # (L, 2)
            tgt_idxs = tgt['gt_lbl_obj'].long()                                                                  # (M,)

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[targets class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_obj = -out_prob[:, tgt_idxs]                                                                    # (L, M)

            # Compute the attribute cost.
            # Contrary to the loss, we don't use the NLL
            out_prob = logits_cls[i].softmax(-1)                                                                 # (L, C)
            tgt_idxs = tgt['gt_lbl_cls'].long().clone()                                                          # (M,)
            tgt_mask = tgt_idxs == 255                                                                           # (M,)
            tgt_idxs[tgt_mask] = 0                                                                               # (M,)
            cost_cls = -out_prob[:, tgt_idxs] * (1 - tgt_mask.float())                                           # (L, M)

            # Compute the msk cost
            out_msks_bev = msks_bev[i]                                                                           # (L, H, W)
            tgt_msks_bev = tgt['gt_msk_bev']                                                                     # (M, H, W)
            out_msks_bev = out_msks_bev.unsqueeze(1).repeat(1, tgt_msks_bev.size(0), 1, 1)                       # (L, M, H, W)
            tgt_msks_bev = tgt_msks_bev.unsqueeze(0).repeat(out_msks_bev.size(0), 1, 1, 1)                       # (L, M, H, W)

            # focal loss
            cost_focal_bev = sigmoid_focal_loss(
                out_msks_bev, tgt_msks_bev, alpha=0.25, gamma=2.0, reduction='none')                             # (L, M, H, W)
            cost_focal_bev = cost_focal_bev.mean(dim=(2, 3))                                                     # (L, M)

            # dice loss
            out_msks_bev = out_msks_bev.flatten(0, 1).sigmoid()                                                  # (L * M, H, W)
            tgt_msks_bev = tgt_msks_bev.flatten(0, 1)                                                            # (L * M, H, W)

            cost_iou_bev = self.dice_loss(out_msks_bev, tgt_msks_bev)                                            # (L * M,)
            cost_iou_bev = cost_iou_bev.view(cost_focal_bev.shape)                                               # (L, M)

            cost_msk_bev = cost_focal_bev + cost_iou_bev

            if self.with_2d:
                out_msks_img = msks_img[i]                                                                       # (L, H, W)
                tgt_msks_img = tgt['gt_msk_img']                                                                 # (M, H, W)
                out_msks_img = out_msks_img.unsqueeze(1).repeat(1, tgt_msks_img.size(0), 1, 1)                   # (L, M, H, W)
                tgt_msks_img = tgt_msks_img.unsqueeze(0).repeat(out_msks_img.size(0), 1, 1, 1)                   # (L, M, H, W)

                # focal loss
                cost_focal_img = sigmoid_focal_loss(
                    out_msks_img, tgt_msks_img, alpha=0.25, gamma=2.0, reduction='none')                         # (L, M, H, W)
                cost_focal_img = cost_focal_img.mean(dim=(2, 3))                                                 # (L, M)

                # dice loss
                out_msks_img = out_msks_img.flatten(0, 1).sigmoid()                                              # (L * M, H, W)
                tgt_msks_img = tgt_msks_img.flatten(0, 1)                                                        # (L * M, H, W)

                cost_iou_img = self.dice_loss(out_msks_img, tgt_msks_img)                                        # (L * M,)
                cost_iou_img = cost_iou_img.view(cost_focal_img.shape)                                           # (L, M)

                cost_msk_img = cost_focal_img + cost_iou_img
            else:
                cost_msk_img = 0.0

            cost_msk = cost_msk_bev + cost_msk_img                                                               # (L, M)

            # Compute the reg cost
            out_regs_bev = regs_bev[i]                                                                           # (L, 3, H, W)
            tgt_regs_bev = tgt['gt_reg_bev']                                                                     # (M, 3, H, W)
            tgt_msks_bev = tgt['gt_msk_bev']                                                                     # (M, H, W)
            out_regs_bev = out_regs_bev.unsqueeze(1).repeat(1, tgt_regs_bev.size(0), 1, 1, 1)                    # (L, M, 3, H, W)
            tgt_regs_bev = tgt_regs_bev.unsqueeze(0).repeat(out_regs_bev.size(0), 1, 1, 1, 1)                    # (L, M, 3, H, W)

            cost_reg_bev = F.l1_loss(out_regs_bev, tgt_regs_bev, reduction='none')                               # (L, M, 3, H, W)
            cost_reg_bev = cost_reg_bev.sum(dim=2)                                                               # (L, M, H, W)
            cost_reg_bev = cost_reg_bev * tgt_msks_bev                                                           # (L, M, H, W)
            cost_reg_bev = cost_reg_bev.sum(dim=(2, 3)) / tgt_msks_bev.sum(dim=(1, 2)).clamp(min=1.0)            # (L, M)

            if self.with_2d:
                out_regs_img = regs_img[i]                                                                       # (L, 3, H, W)
                tgt_regs_img = tgt['gt_reg_img']                                                                 # (M, 3, H, W)
                tgt_msks_img = tgt['gt_msk_img']                                                                 # (M, H, W)
                out_regs_img = out_regs_img.unsqueeze(1).repeat(1, tgt_regs_img.size(0), 1, 1, 1)                # (L, M, 3, H, W)
                tgt_regs_img = tgt_regs_img.unsqueeze(0).repeat(out_regs_img.size(0), 1, 1, 1, 1)                # (L, M, 3, H, W)

                # cost_reg_img = F.l1_loss(out_regs_img, tgt_regs_img, reduction='none')                           # (L, M, 3, H, W)
                cost_reg_img = F.l1_loss(out_regs_img[:, :, :2], tgt_regs_img[:, :, :2], reduction='none')       # (L, M, 2, H, W)
                cost_reg_img = cost_reg_img.sum(dim=2)                                                           # (L, M, H, W)
                cost_reg_img = cost_reg_img * tgt_msks_img                                                       # (L, M, H, W)
                cost_reg_img = cost_reg_img.sum(dim=(2, 3)) / tgt_msks_img.sum(dim=(1, 2)).clamp(min=1.0)        # (L, M)
            else:
                cost_reg_img = 0.0

            cost_reg = cost_reg_bev + cost_reg_img                                                               # (L, M)

            # Final cost matrix
            C = self.obj_weight * cost_obj + self.cls_weight * cost_cls + \
                self.msk_weight * cost_msk + self.reg_weight * cost_reg

            C = C.view(num_queries, -1).cpu()
            C_list.append(C)

        indices = [linear_sum_assignment(c) for c in C_list]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
