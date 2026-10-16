# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, obj_weight: float = 1, cls_weight: float = 1,  loc_weight: float = 1, reg_weight: float = 1,
                 rng_weight: float = 1, line_width: int = 13, with_2d: bool = True):
        """Creates the matcher
        """
        super().__init__()
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.reg_weight = reg_weight
        self.rng_weight = rng_weight
        self.line_width = line_width
        self.with_2d    = with_2d

    @torch.no_grad()
    def forward(self, logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img,
                ranges_bev, ranges_img, targets):
        """ Performs the matching
        """
        bs, num_queries = logits_obj.shape[:2]
        device = logits_obj.device

        row_msks_bev = masks_bev.softmax(dim=3)                                                                # (N, L, H, W)
        row_locs_bev = torch.arange(0, row_msks_bev.size(3), dtype=torch.float32, device=device)               # (W,)
        row_locs_bev = (row_locs_bev * row_msks_bev).sum(dim=3)                                                # (N, L, H)
        row_rgxs_bev = regs_bev[:, 0::2]                                                                       # (N, L, H, W)
        row_rgzs_bev = regs_bev[:, 1::2]                                                                       # (N, L, H, W)

        if self.with_2d:
            row_msks_img = masks_img.softmax(dim=3)                                                            # (N, L, H, W)
            row_locs_img = torch.arange(0, row_msks_img.size(3), dtype=torch.float32, device=device)           # (W,)
            row_locs_img = (row_locs_img * row_msks_img).sum(dim=3)                                            # (N, L, H)
            row_rgxs_img = regs_img[:, 0::2]                                                                   # (N, L, H, W)
            row_rgzs_img = regs_img[:, 1::2]                                                                   # (N, L, H, W)

        C_list = []
        for i, tgt in enumerate(targets):
            # We flatten to compute the cost matrices in a batch
            out_prob = logits_obj[i].softmax(-1)                                                               # (L, 2)
            tgt_idxs = tgt['gt_label_obj'].long()                                                              # (M,)

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[targets class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_obj = -out_prob[:, tgt_idxs]                                                                  # (L, M)

            # Compute the attribute cost.
            # Contrary to the loss, we don't use the NLL
            out_prob = logits_cls[i].softmax(-1)                                                               # (L, C)
            tgt_idxs = tgt['gt_label_cls'].long()                                                              # (M,)
            tgt_mask = tgt_idxs == 255                                                                         # (M,)
            tgt_idxs[tgt_mask] = 0                                                                             # (M,)
            cost_cls = -out_prob[:, tgt_idxs] * (1 - tgt_mask.float())                                         # (L, M)

            # Compute the location cost.
            out_row_locs_bev = row_locs_bev[i]                                                                 # (L, H)
            tgt_row_locs_bev = tgt['gt_row_loc_bev']                                                           # (M, H)
            tgt_loc_msks_bev = tgt['gt_loc_msk_bev']                                                           # (M, H)

            # Loc loss
            out_row_locs_bev = out_row_locs_bev.unsqueeze(1).repeat(1, tgt_row_locs_bev.size(0), 1)            # (L, M, H)
            tgt_row_locs_bev = tgt_row_locs_bev.unsqueeze(0).repeat(out_row_locs_bev.size(0), 1, 1)            # (L, M, H)
            cost_row_loc_bev = F.l1_loss(out_row_locs_bev, tgt_row_locs_bev, reduction='none')                 # (L, M, H)
            cost_row_loc_bev = cost_row_loc_bev * tgt_loc_msks_bev                                             # (L, M, H)
            cost_row_loc_bev = cost_row_loc_bev.sum(dim=2) / tgt_loc_msks_bev.sum(dim=1).clamp(min=1.0)        # (L, M)

            cost_row_loc = cost_row_loc_bev                                                                    # (L, M)

            # IoU loss
            out_row_loc1_bev = out_row_locs_bev - self.line_width                                              # (L, M, H)
            out_row_loc2_bev = out_row_locs_bev + self.line_width                                              # (L, M, H)
            tgt_row_loc1_bev = tgt_row_locs_bev - self.line_width                                              # (L, M, H)
            tgt_row_loc2_bev = tgt_row_locs_bev + self.line_width                                              # (L, M, H)

            line_row_ovr_bev = (torch.min(out_row_loc2_bev, tgt_row_loc2_bev) -
                                torch.max(out_row_loc1_bev, tgt_row_loc1_bev)) * tgt_loc_msks_bev              # (L, M, H)
            line_row_uni_bev = (torch.max(out_row_loc2_bev, tgt_row_loc2_bev) -
                                torch.min(out_row_loc1_bev, tgt_row_loc1_bev)) * tgt_loc_msks_bev              # (L, M, H)
            line_row_iou_bev = line_row_ovr_bev.sum(dim=2) / (line_row_uni_bev.sum(dim=2) + 1e-9)              # (L, M)
            cost_row_iou_bev = 1 - line_row_iou_bev                                                            # (L, M)

            cost_row_iou = cost_row_iou_bev                                                                    # (L, M)

            if self.with_2d:
                # Loc loss
                out_row_locs_img = row_locs_img[i]                                                             # (L, H)
                tgt_row_locs_img = tgt['gt_row_loc_img']                                                       # (M, H)
                tgt_loc_msks_img = tgt['gt_loc_msk_img']                                                       # (M, H)

                out_row_locs_img = out_row_locs_img.unsqueeze(1).repeat(1, tgt_row_locs_img.size(0), 1)        # (L, M, H)
                tgt_row_locs_img = tgt_row_locs_img.unsqueeze(0).repeat(out_row_locs_img.size(0), 1, 1)        # (L, M, H)
                cost_row_loc_img = F.l1_loss(out_row_locs_img, tgt_row_locs_img, reduction='none')             # (L, M, H)
                cost_row_loc_img = cost_row_loc_img * tgt_loc_msks_img                                         # (L, M, H)
                cost_row_loc_img = cost_row_loc_img.sum(dim=2) / tgt_loc_msks_img.sum(dim=1).clamp(min=1.0)    # (L, M)

                cost_row_loc = cost_row_loc + cost_row_loc_img                                                 # (L, M)

                # IoU loss
                out_row_loc1_img = out_row_locs_img - self.line_width                                          # (L, M, H)
                out_row_loc2_img = out_row_locs_img + self.line_width                                          # (L, M, H)
                tgt_row_loc1_img = tgt_row_locs_img - self.line_width                                          # (L, M, H)
                tgt_row_loc2_img = tgt_row_locs_img + self.line_width                                          # (L, M, H)

                line_row_ovr_img = (torch.min(out_row_loc2_img, tgt_row_loc2_img) -
                                    torch.max(out_row_loc1_img, tgt_row_loc1_img)) * tgt_loc_msks_img          # (L, M, H)
                line_row_uni_img = (torch.max(out_row_loc2_img, tgt_row_loc2_img) -
                                    torch.min(out_row_loc1_img, tgt_row_loc1_img)) * tgt_loc_msks_img          # (L, M, H)
                line_row_iou_img = line_row_ovr_img.sum(dim=2) / (line_row_uni_img.sum(dim=2) + 1e-9)          # (L, M)
                cost_row_iou_img = 1 - line_row_iou_img                                                        # (L, M)

                cost_row_iou = cost_row_iou + cost_row_iou_img                                                 # (L, M)

            cost_loc = cost_row_loc + cost_row_iou * 2.0                                                       # (L, M)

            # out_row_msks = row_msks[i]                                                                       # (L, H, W)
            # out_row_msks = out_row_msks.unsqueeze(1).repeat(1, tgt_row_locs.size(1), 1, 1)                   # (L, M, H, W)
            # cost_row_cls = F.cross_entropy(out_row_msks.permute(0, 3, 1, 2),
            #                                tgt_row_locs.long(), reduction='none')                            # (L, M, H)
            # cost_row_cls = cost_row_cls * tgt_row_msks                                                       # (L, M, H)
            # cost_row_cls = cost_row_cls.sum(dim=2) / tgt_row_msks.sum(dim=1).clamp(min=1.0)                  # (L, M)
            #
            # cost_loc = cost_row_loc + cost_row_iou * 2.0 + cost_row_cls * 5.0                                # (L, M)

            # Compute the regression loss
            out_row_rgxs_bev = row_rgxs_bev[i]                                                                 # (L, H, W)
            out_row_rgzs_bev = row_rgzs_bev[i]                                                                 # (L, H, W)
            tgt_row_rgxs_bev = tgt['gt_row_rgx_bev']                                                           # (M, H, W)
            tgt_row_rgzs_bev = tgt['gt_row_rgz_bev']                                                           # (M, H, W)
            tgt_reg_msks_bev = tgt['gt_reg_msk_bev']                                                           # (M, H, W)

            out_row_rgxs_bev = out_row_rgxs_bev.unsqueeze(1).repeat(1, tgt_row_rgxs_bev.size(0), 1, 1)         # (L, M, H, W)
            tgt_row_rgxs_bev = tgt_row_rgxs_bev.unsqueeze(0).repeat(out_row_rgxs_bev.size(0), 1, 1, 1)         # (L, M, H, W)
            cost_row_rgx_bev = F.l1_loss(out_row_rgxs_bev, tgt_row_rgxs_bev, reduction='none')                 # (L, M, H, W)

            # out_row_rgzs_bev = out_row_rgzs_bev.unsqueeze(1).repeat(1, tgt_row_rgzs_bev.size(0), 1, 1)         # (L, M, H, W)
            # tgt_row_rgzs_bev = tgt_row_rgzs_bev.unsqueeze(0).repeat(out_row_rgzs_bev.size(0), 1, 1, 1)         # (L, M, H, W)
            # cost_row_rgz_bev = F.l1_loss(out_row_rgzs_bev, tgt_row_rgzs_bev, reduction='none')                 # (L, M, H, W)
            cost_row_rgz_bev = 0.0
            cost_row_reg_bev = (cost_row_rgx_bev + cost_row_rgz_bev) *  tgt_reg_msks_bev                       # (L, M, H, W)
            cost_row_reg_bev = cost_row_reg_bev.sum(dim=(2, 3)) / tgt_reg_msks_bev.sum(dim=(1, 2)).clamp(min=1.0)  # (L, M)

            cost_row_reg = cost_row_reg_bev                                                                    # (L, M)

            if self.with_2d:
                out_row_rgxs_img = row_rgxs_img[i]                                                             # (L, H, W)
                out_row_rgzs_img = row_rgzs_img[i]                                                             # (L, H, W)
                tgt_row_rgxs_img = tgt['gt_row_rgx_img']                                                       # (M, H, W)
                tgt_row_rgzs_img = tgt['gt_row_rgz_img']                                                       # (M, H, W)
                tgt_reg_msks_img = tgt['gt_reg_msk_img']                                                       # (M, H, W)

                out_row_rgxs_img = out_row_rgxs_img.unsqueeze(1).repeat(1, tgt_row_rgxs_img.size(0), 1, 1)     # (L, M, H, W)
                tgt_row_rgxs_img = tgt_row_rgxs_img.unsqueeze(0).repeat(out_row_rgxs_img.size(0), 1, 1, 1)     # (L, M, H, W)
                cost_row_rgx_img = F.l1_loss(out_row_rgxs_img, tgt_row_rgxs_img, reduction='none')             # (L, M, H, W)

                # out_row_rgzs_img = out_row_rgzs_img.unsqueeze(1).repeat(1, tgt_row_rgzs_img.size(0), 1, 1)     # (L, M, H, W)
                # tgt_row_rgzs_img = tgt_row_rgzs_img.unsqueeze(0).repeat(out_row_rgzs_img.size(0), 1, 1, 1)     # (L, M, H, W)
                # cost_row_rgz_img = F.l1_loss(out_row_rgzs_img, tgt_row_rgzs_img, reduction='none')             # (L, M, H, W)
                cost_row_rgz_img = 0.0
                cost_row_reg_img = (cost_row_rgx_img + cost_row_rgz_img) *  tgt_reg_msks_img                   # (L, M, H, W)
                cost_row_reg_img = cost_row_reg_img.sum(dim=(2, 3)) / tgt_reg_msks_img.sum(dim=(1, 2)).clamp(min=1.0)  # (L, M)

                cost_row_reg = cost_row_reg + cost_row_reg_img                                                 # (L, M)

            cost_reg = cost_row_reg

            # Compute the range cost.
            # lane_ranges: (N, L, 2 * 2)
            out_row_rngs_bev = ranges_bev[i]                                                                   # (L, 2)
            tgt_row_rngs_bev = tgt['gt_row_rng_bev']                                                           # (M, 2)
            out_row_rngs_bev = out_row_rngs_bev.unsqueeze(1).repeat(1, tgt_row_rngs_bev.size(0), 1)            # (L, M, 2)
            tgt_row_rngs_bev = tgt_row_rngs_bev.unsqueeze(0).repeat(out_row_rngs_bev.size(0), 1, 1)            # (L, M, 2)
            cost_row_rng_bev = F.l1_loss(out_row_rngs_bev, tgt_row_rngs_bev, reduction='none')                 # (L, M, 2)
            cost_row_rng_bev = cost_row_rng_bev.sum(dim=2)                                                     # (L, M)

            cost_row_rng = cost_row_rng_bev                                                                    # (L, M)

            if self.with_2d:
                out_row_rngs_img = ranges_img[i]                                                               # (L, 2)
                tgt_row_rngs_img = tgt['gt_row_rng_img']                                                       # (M, 2)
                out_row_rngs_img = out_row_rngs_img.unsqueeze(1).repeat(1, tgt_row_rngs_img.size(0), 1)        # (L, M, 2)
                tgt_row_rngs_img = tgt_row_rngs_img.unsqueeze(0).repeat(out_row_rngs_img.size(0), 1, 1)        # (L, M, 2)
                cost_row_rng_img = F.l1_loss(out_row_rngs_img, tgt_row_rngs_img, reduction='none')             # (L, M, 2)
                cost_row_rng_img = cost_row_rng_img.sum(dim=2)                                                 # (L, M)

                cost_row_rng = cost_row_rng + cost_row_rng_img                                                 # (L, M)

            cost_rng = cost_row_rng                                                                            # (L, M)

            # Final cost matrix
            C = self.obj_weight * cost_obj + self.cls_weight * cost_cls + self.loc_weight * cost_loc + \
                self.reg_weight * cost_reg + self.rng_weight * cost_rng                                        # (L, M)

            C = C.view(num_queries, -1).cpu()
            C_list.append(C)
        indices = [linear_sum_assignment(c) for c in C_list]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
