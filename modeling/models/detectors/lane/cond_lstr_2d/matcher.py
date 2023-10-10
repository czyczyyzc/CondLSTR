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
                 rng_weight: float = 1, line_width=13):
        """Creates the matcher
        """
        super().__init__()
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.reg_weight = reg_weight
        self.rng_weight = rng_weight
        self.line_width = line_width

    @torch.no_grad()
    def forward(self, logits_obj, logits_cls, regs, masks, lane_ranges, targets):
        """ Performs the matching
        """
        bs, num_queries = logits_obj.shape[:2]
        device = logits_obj.device

        row_msks = masks.softmax(dim=3)                                                                # (N, L, H, W)
        row_locs = torch.arange(0, row_msks.size(3), dtype=torch.float32, device=device)               # (W,)
        row_locs = (row_locs * row_msks).sum(dim=3)                                                    # (N, L, H)

        C_list = []
        for i, tgt in enumerate(targets):
            # We flatten to compute the cost matrices in a batch
            out_prob = logits_obj[i].softmax(-1)                                                       # (L, 2)
            tgt_idxs = tgt['gt_label_obj'].long()                                                      # (M,)

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[targets class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_obj = -out_prob[:, tgt_idxs]                                                          # (L, M)

            # Compute the attribute cost.
            # logits_cls: (N, L, 21), Contrary to the loss, we don't use the NLL
            out_prob = logits_cls[i].softmax(-1)                                                       # (L, C)
            tgt_idxs = tgt['gt_label_cls'].long()                                                      # (M,)
            tgt_mask = tgt_idxs == 255                                                                 # (M,)
            tgt_idxs[tgt_mask] = 0                                                                     # (M,)
            cost_cls = -out_prob[:, tgt_idxs] * (1 - tgt_mask.float())                                 # (L, M)

            # Compute the location cost.
            out_row_locs = row_locs[i]                                                                 # (L, H)
            tgt_row_locs = tgt['gt_row_loc']                                                           # (M, H)
            tgt_row_msks = tgt['gt_row_loc_mask']                                                      # (M, H)

            # Loc loss
            out_row_locs = out_row_locs.unsqueeze(1).repeat(1, tgt_row_locs.size(0), 1)                # (L, M, H)
            tgt_row_locs = tgt_row_locs.unsqueeze(0).repeat(out_row_locs.size(0), 1, 1)                # (L, M, H)
            cost_row_loc = F.l1_loss(out_row_locs, tgt_row_locs, reduction='none')                     # (L, M, H)
            cost_row_loc = cost_row_loc * tgt_row_msks                                                 # (L, M, H)
            cost_row_loc = cost_row_loc.sum(dim=2) / tgt_row_msks.sum(dim=1).clamp(min=1.0)            # (L, M)

            # IoU loss
            out_row_loc1 = out_row_locs - self.line_width                                              # (L, M, H)
            out_row_loc2 = out_row_locs + self.line_width                                              # (L, M, H)
            tgt_row_loc1 = tgt_row_locs - self.line_width                                              # (L, M, H)
            tgt_row_loc2 = tgt_row_locs + self.line_width                                              # (L, M, H)

            line_row_ovr = (torch.min(out_row_loc2, tgt_row_loc2) -
                            torch.max(out_row_loc1, tgt_row_loc1)) * tgt_row_msks                      # (L, M, H)
            line_row_uni = (torch.max(out_row_loc2, tgt_row_loc2) -
                            torch.min(out_row_loc1, tgt_row_loc1)) * tgt_row_msks                      # (L, M, H)
            line_row_iou = line_row_ovr.sum(dim=2) / (line_row_uni.sum(dim=2) + 1e-9)                  # (L, M)
            cost_row_iou = 1 - line_row_iou

            cost_loc = cost_row_loc + cost_row_iou * 2.0

            # out_row_msks = row_msks[i]                                                                 # (L, H, W)
            # out_row_msks = out_row_msks.unsqueeze(1).repeat(1, tgt_row_locs.size(1), 1, 1)             # (L, M, H, W)
            # cost_row_cls = F.cross_entropy(out_row_msks.permute(0, 3, 1, 2),
            #                                tgt_row_locs.long(), reduction='none')                      # (L, M, H)
            # cost_row_cls = cost_row_cls * tgt_row_msks                                                 # (L, M, H)
            # cost_row_cls = cost_row_cls.sum(dim=2) / tgt_row_msks.sum(dim=1).clamp(min=1.0)            # (L, M)
            #
            # cost_loc = cost_row_loc + cost_row_iou * 2.0 + cost_row_cls * 5.0                          # (L, M)

            # Compute the regression loss
            out_row_regs = regs[i]                                                                     # (L, H, W)
            tgt_row_regs = tgt['gt_row_reg']                                                           # (M, H, W)
            tgt_row_msks = tgt['gt_row_reg_mask']                                                      # (M, H, W)

            out_row_regs = out_row_regs.unsqueeze(1).repeat(1, tgt_row_regs.size(0), 1, 1)             # (L, M, H, W)
            tgt_row_regs = tgt_row_regs.unsqueeze(0).repeat(out_row_regs.size(0), 1, 1, 1)             # (L, M, H, W)
            cost_row_reg = F.l1_loss(out_row_regs, tgt_row_regs, reduction='none')                     # (L, M, H, W)
            cost_row_reg = cost_row_reg * tgt_row_msks                                                 # (L, M, H, W)
            cost_row_reg = cost_row_reg.sum(dim=(2, 3)) / tgt_row_msks.sum(dim=(1, 2)).clamp(min=1.0)  # (L, M)

            cost_reg = cost_row_reg

            # Compute the range cost.
            # lane_ranges: (N, L, 2 * 2)
            out_row_rngs = lane_ranges[i]                                                              # (L, 2)
            tgt_row_rngs = tgt['gt_row_rng']                                                           # (M, 2)
            out_row_rngs = out_row_rngs.unsqueeze(1).repeat(1, tgt_row_rngs.size(0), 1)                # (L, M, 2)
            tgt_row_rngs = tgt_row_rngs.unsqueeze(0).repeat(out_row_rngs.size(0), 1, 1)                # (L, M, 2)
            cost_row_rng = F.l1_loss(out_row_rngs, tgt_row_rngs, reduction='none')                     # (L, M, 2)
            cost_row_rng = cost_row_rng.sum(dim=2)                                                     # (L, M)

            cost_rng = cost_row_rng                                                                    # (L, M)

            # Final cost matrix
            C = self.obj_weight * cost_obj + self.cls_weight * cost_cls + self.loc_weight * cost_loc + \
                self.reg_weight * cost_reg + self.rng_weight * cost_rng                                # (L, M)
            C_list.append(C.cpu())                                                                     # (L, M)

        # sizes = [tgt['gt_label_obj'].shape[0] for tgt in targets]                                    # [M1, M2]
        # # C.split(sizes, -1): [(N, L, M1), (N, L, M2)]
        # # c[i] (L, Mi)
        # # indices: [(L, L), (L, L), ...]
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        indices = [linear_sum_assignment(c) for c in C_list]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
