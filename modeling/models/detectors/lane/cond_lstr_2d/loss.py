import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .matcher import HungarianMatcher
from modeling.models.utils import is_dist_avail_and_initialized, get_world_size, accuracy


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, eos_coef=1., obj_weight=1., cls_weight=1., loc_weight=1.,
                 reg_weight=1., rng_weight=1., line_width=15):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            eos_coef: relative classification weight applied to the no-object category
            matcher: module able to compute a matching between targets and proposals
        """
        super().__init__()
        self.matcher    = matcher
        self.eos_coef   = eos_coef
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.reg_weight = reg_weight
        self.rng_weight = rng_weight
        self.line_width = line_width

        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_obj(self, logits, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        device = logits.device
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([tgt['gt_label_obj'][J].long() for tgt, (_, J) in zip(targets, indices)])  # (M1 + M2,)
        target_classes = torch.full(logits.shape[:2], 1, dtype=torch.int64, device=device)                      # (N, L)
        target_classes[idx] = target_classes_o                                                                  # (M1 + M2,)
        loss_obj = F.cross_entropy(logits.transpose(1, 2), target_classes, self.empty_weight)      # (N, C, L) (N, L)
        return loss_obj

    def loss_cls(self, logits, targets, indices, num_lanes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # logits: (N, L, 20)
        idx = self._get_src_permutation_idx(indices)
        src_logits = logits[idx]                                                                   # (M1 + M2, 20)
        tgt_labels = torch.cat([tgt['gt_label_cls'][i] for tgt, (_, i) in zip(targets, indices)])  # (M1 + M2,)
        class_weight = targets[0].get('class_weight', None)
        
        # num_category = src_logits.shape[-1]
        # gt_category = tgt_labels.clone()
        # gt_category[gt_category == 255] = 0
        # gt_category_onehot = F.one_hot(gt_category, num_classes=num_category).float()
        # weight_category = 1.0 - torch.sum(gt_category_onehot.reshape(-1, num_category), dim=0) / torch.sum(gt_category_onehot)
        # loss_cls = F.cross_entropy(src_logits, tgt_labels, weight=weight_category, ignore_index=255, reduction='sum')

        loss_cls = F.cross_entropy(src_logits, tgt_labels, weight=class_weight, ignore_index=255, reduction='sum')
        loss_cls = loss_cls / num_lanes
        return loss_cls

    @torch.no_grad()
    def loss_cardinality(self, logits, targets):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        device = logits.device
        tgt_lengths = torch.as_tensor([tgt['gt_label'].shape[0] for tgt in targets], device=device)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)                                       # (N,)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return card_err

    def loss_loc(self, masks, targets, indices, num_lanes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The targets boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        device = masks.device
        # masks: (N, L, H, W)
        row_msks = masks.softmax(dim=3)                                                                  # (N, L, H, W)

        row_locs = torch.arange(0, row_msks.size(3), dtype=torch.float32, device=device)                 # (W,)
        row_locs = (row_locs * row_msks).sum(dim=3)                                                      # (N, L, H)

        idx = self._get_src_permutation_idx(indices)

        src_row_locs = row_locs[idx]                                                                     # (M1 + M2, H)
        tgt_row_locs = torch.cat([tgt['gt_row_loc'][i] for tgt, (_, i) in zip(targets, indices)])        # (M1 + M2, H)
        tgt_row_msks = torch.cat([tgt['gt_row_loc_mask'][i] for tgt, (_, i) in zip(targets, indices)])   # (M1 + M2, H)

        loss_row_loc = F.l1_loss(src_row_locs, tgt_row_locs, reduction='none')                           # (M1 + M2, H)
        loss_row_loc = loss_row_loc * tgt_row_msks                                                       # (M1 + M2, H)
        loss_row_loc = loss_row_loc.sum(dim=1) / tgt_row_msks.sum(dim=1).clamp(min=1.0)                  # (M1 + M2)
        loss_row_loc = loss_row_loc.sum() / num_lanes

        src_row_loc1 = src_row_locs - self.line_width                                                    # (M1 + M2, H)
        src_row_loc2 = src_row_locs + self.line_width                                                    # (M1 + M2, H)
        tgt_row_loc1 = tgt_row_locs - self.line_width                                                    # (M1 + M2, H)
        tgt_row_loc2 = tgt_row_locs + self.line_width                                                    # (M1 + M2, H)

        line_row_ovr = (torch.min(src_row_loc2, tgt_row_loc2) -
                        torch.max(src_row_loc1, tgt_row_loc1)) * tgt_row_msks                            # (M1 + M2, H)
        line_row_uni = (torch.max(src_row_loc2, tgt_row_loc2) -
                        torch.min(src_row_loc1, tgt_row_loc1)) * tgt_row_msks                            # (M1 + M2, H)
        line_row_iou = line_row_ovr.sum(dim=1) / (line_row_uni.sum(dim=1) + 1e-9)                        # (M1 + M2)
        loss_row_iou = 1 - line_row_iou                                                                  # (M1 + M2)
        loss_row_iou = loss_row_iou.sum() / num_lanes

        loss_loc = loss_row_loc + loss_row_iou * 2.0

        # src_row_msks = masks[idx]
        # loss_row_cls = F.cross_entropy(src_row_msks.permute(0, 2, 1), tgt_row_locs.long(), reduction='none')  # (M1 + M2, H)
        # loss_row_cls = loss_row_cls * tgt_row_msks                                                       # (M1 + M2, H)
        # loss_row_cls = loss_row_cls.sum(dim=1) / tgt_row_msks.sum(dim=1).clamp(min=1.0)                  # (M1 + M2)
        # loss_row_cls = loss_row_cls.sum() / num_lanes
        #
        # loss_loc = loss_row_loc + loss_row_iou * 2.0 + loss_row_cls * 5.0
        return loss_loc

    def loss_reg(self, regs, targets, indices, num_lanes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The targets boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # regs: (N, L, H, W)
        idx = self._get_src_permutation_idx(indices)
        row_regs = regs

        src_row_regs = row_regs[idx]                                                                    # (M1 + M2, H, W)
        tgt_row_regs = torch.cat([tgt['gt_row_reg'][i] for tgt, (_, i) in zip(targets, indices)])       # (M1 + M2, H, W)
        tgt_row_msks = torch.cat([tgt['gt_row_reg_mask'][i] for tgt, (_, i) in zip(targets, indices)])  # (M1 + M2, H, W)

        loss_row_reg = F.l1_loss(src_row_regs, tgt_row_regs, reduction='none')                          # (M1 + M2, H, W)
        loss_row_reg = loss_row_reg * tgt_row_msks                                                      # (M1 + M2, H, W)
        loss_row_reg = loss_row_reg.sum(dim=(1, 2)) / tgt_row_msks.sum(dim=(1, 2)).clamp(min=1.0)       # (M1 + M2)
        loss_row_reg = loss_row_reg.sum() / num_lanes

        loss_reg = loss_row_reg
        return loss_reg

    def loss_rng(self, lane_ranges, targets, indices, num_lanes):
        # lane_ranges: (N, L, 2)
        row_rngs = lane_ranges                                                                          # (N, L, 2)

        idx = self._get_src_permutation_idx(indices)

        src_row_rngs = row_rngs[idx]                                                                    # (M1 + M2, 2)
        tgt_row_rngs = torch.cat([tgt['gt_row_rng'][i] for tgt, (_, i) in zip(targets, indices)])       # (M1 + M2, 2)

        loss_row_rng = F.l1_loss(src_row_rngs, tgt_row_rngs, reduction='sum')
        loss_row_rng = loss_row_rng / num_lanes

        loss_rng = loss_row_rng
        return loss_rng

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])       # (N * L,)
        src_idx = torch.cat([src for (src, _) in indices])                                           # (N * L,)
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])       # (N * L,)
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])                                           # (N * L,)
        return batch_idx, tgt_idx

    def forward(self, logits_obj, logits_cls, regs, masks, lane_ranges, targets, num_lanes):
        """ This performs the loss computation.
        Parameters:
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        device = logits_obj.device
        indices = self.matcher(logits_obj, logits_cls, regs, masks, lane_ranges, targets)

        # Compute all the requested losses
        loss_obj = self.loss_obj(logits_obj, targets, indices)
        loss_cls = self.loss_cls(logits_cls, targets, indices, num_lanes)
        loss_loc = self.loss_loc(masks, targets, indices, num_lanes)
        loss_reg = self.loss_reg(regs, targets, indices, num_lanes)
        loss_rng = self.loss_rng(lane_ranges, targets, indices, num_lanes)

        loss_obj = loss_obj * self.obj_weight
        loss_cls = loss_cls * self.cls_weight
        loss_loc = loss_loc * self.loc_weight
        loss_reg = loss_reg * self.reg_weight
        loss_rng = loss_rng * self.rng_weight
        return loss_obj, loss_cls, loss_loc, loss_reg, loss_rng


class CondLSTR2DLoss(nn.Module):

    def __init__(self, min_points=10, line_width=15, weight_dict=None, eos_coef=1.):
        """
        Args:
            weights is a dict which sets the weight of the loss
            eg. {hm_weight: 1, kp_weight: 1, ins_weight: 1}
        """
        super(CondLSTR2DLoss, self).__init__()
        self.min_points = min_points
        self.line_width = line_width

        if weight_dict is None:
            weight_dict = {'obj_weight': 1.0, 'cls_weight': 1.0, 'loc_weight': 1.0, 'reg_weight': 1.0, 'rng_weight': 5.0}

        matcher = HungarianMatcher(obj_weight=weight_dict['obj_weight'],
                                   cls_weight=weight_dict['cls_weight'],
                                   loc_weight=weight_dict['loc_weight'],
                                   reg_weight=weight_dict['reg_weight'],
                                   rng_weight=weight_dict['rng_weight'],
                                   line_width=line_width)

        self.criterion = SetCriterion(matcher=matcher,
                                      eos_coef=eos_coef,
                                      obj_weight=weight_dict['obj_weight'],
                                      cls_weight=weight_dict['cls_weight'],
                                      loc_weight=weight_dict['loc_weight'],
                                      reg_weight=weight_dict['reg_weight'],
                                      rng_weight=weight_dict['rng_weight'],
                                      line_width=line_width)

    @torch.no_grad()
    def preprocess(self, gt_masks, gt_labels, mask_h, mask_w, img_metas):
        # gt_masks: (N, H, W)
        if gt_labels is None:
            gt_labels = [None for _ in range(len(gt_masks))]
        
        targets = []
        for gt_mask, gt_attr, img_meta in zip(gt_masks, gt_labels, img_metas):
            device = gt_mask.device
            if len(gt_mask) > 0:
                gt_mask = gt_mask[:, None].float()                                                        # (M, 1, H, W)
                gt_mask = F.interpolate(gt_mask, size=(mask_h, mask_w), mode='nearest')[:, 0]             # (M, H, W)
            else:
                gt_mask = torch.zeros((0, mask_h, mask_w), dtype=torch.float32, device=device)            # (0, H, W)
            
            if gt_attr is None:
                gt_attr = torch.zeros(len(gt_mask), dtype=torch.int64, device=device)                     # (M,)
            else:
                gt_attr = gt_attr.long()                                                                  # (M,)
            assert len(gt_mask) == len(gt_attr)
            
            img_h, img_w = img_meta['img_shape'][:2]
            inp_h, inp_w = img_meta['input_shape'][:2]
            img_h, img_w = int(img_h * mask_h / inp_h), int(img_w * mask_w / inp_w)

            if len(gt_mask) > 0:
                gt_keep = (gt_mask.max(dim=2)[0].sum(dim=1) >= self.min_points) | \
                          (gt_mask.max(dim=1)[0].sum(dim=1) >= self.min_points)                           # (M,)
                gt_mask = gt_mask[gt_keep]                                                                # (M, H, W)
                gt_attr = gt_attr[gt_keep]                                                                # (M,)

            if len(gt_mask) > 0:
                row_crd = torch.arange(0, gt_mask.size(2), dtype=torch.float32, device=device)            # (W,)
                row_ptx = (gt_mask * row_crd).sum(dim=2) / gt_mask.sum(dim=2).clamp(min=1.0)              # (M, H)
                row_msk = gt_mask.max(dim=2)[0]                                                           # (M, H)

                row_sta = row_msk.argmax(dim=1)                                                           # (M,)
                row_end = row_msk.flip(1).argmax(dim=1)                                                   # (M,)
                row_end = row_msk.size(1) - 1 - row_end                                                   # (M,)
                row_rng = torch.stack([row_sta, row_end], dim=1) / img_h                                  # (M, 2)

                row_reg = row_ptx[:, :, None] - row_crd                                                   # (M, H, W)
                reg_msk = (row_reg >= -self.line_width * 4) & (row_reg <= self.line_width * 4) & row_msk[:, :, None].bool()
                row_reg[~reg_msk] = 0.0

                row_loc      = row_ptx                                                                    # (M, H)
                row_loc_mask = row_msk                                                                    # (M, H)
                row_reg_mask = reg_msk.float()                                                            # (M, H, W)
                label_obj    = torch.zeros(len(gt_mask), dtype=torch.int64, device=device)                # (M,)
                label_cls    = gt_attr                                                                    # (M,)
            else:
                row_loc      = torch.zeros((0, mask_h), dtype=torch.float32, device=device)
                row_loc_mask = torch.zeros((0, mask_h), dtype=torch.float32, device=device)
                row_reg      = torch.zeros((0, mask_h, mask_w), dtype=torch.float32, device=device)
                row_reg_mask = torch.zeros((0, mask_h, mask_w), dtype=torch.float32, device=device)
                row_rng      = torch.zeros((0, 2), dtype=torch.float32, device=device)
                label_obj    = torch.zeros((0,), dtype=torch.int64, device=device)
                label_cls    = torch.zeros((0,), dtype=torch.int64, device=device)

            target = dict(
                gt_row_rng=row_rng,
                gt_row_loc=row_loc,
                gt_row_reg=row_reg,
                gt_row_loc_mask=row_loc_mask,
                gt_row_reg_mask=row_reg_mask,
                gt_label_obj=label_obj,
                gt_label_cls=label_cls,
            )
            if 'class_weight' in img_meta:
                target.update(class_weight=img_meta['class_weight'])
            targets.append(target)
        return targets

    def forward(self, logits_obj, logits_cls, regs, masks, lane_ranges, gt_masks, gt_labels, img_metas):        
        if isinstance(masks, (list, tuple)):
            mask_h, mask_w = masks[0].size(2), masks[1].size(3)
            device = masks[0].device
        else:
            mask_h, mask_w = masks.size(2), masks.size(3)
            device = masks.device
        targets = self.preprocess(gt_masks, gt_labels, mask_h, mask_w, img_metas)

        # Compute the average number of targets boxes accross all nodes, for normalization purposes
        num_lanes = sum(tgt['gt_label_obj'].shape[0] for tgt in targets)
        num_lanes = torch.as_tensor([num_lanes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            dist.all_reduce(num_lanes)
        num_lanes = torch.clamp(num_lanes / get_world_size(), min=1).item()
        
        if isinstance(logits_obj, (list, tuple)):
            loss_obj, loss_cls, loss_loc, loss_reg, loss_rng = 0, 0, 0, 0, 0
            for logits_obj_, logits_cls_, regs_, masks_, lane_ranges_ in \
                    zip(logits_obj, logits_cls, regs, masks, lane_ranges):
                loss_obj_, loss_cls_, loss_loc_, loss_reg_, loss_rng_ = self.criterion(
                    logits_obj_, logits_cls_, regs_, masks_, lane_ranges_, targets, num_lanes)
                loss_obj += loss_obj_
                loss_cls += loss_cls_
                loss_loc += loss_loc_
                loss_reg += loss_reg_
                loss_rng += loss_rng_
        else:
            loss_obj, loss_cls, loss_loc, loss_reg, loss_rng = self.criterion(
                logits_obj, logits_cls, regs, masks, lane_ranges, targets, num_lanes)
        return loss_obj, loss_cls, loss_loc, loss_reg, loss_rng


# import os
# import cv2
# import numpy as np
# from PIL import Image
#
# path = '/data/sets/czy/1'
# os.makedirs(path, exist_ok=True)
# for j, mask in enumerate(gt_mask):
#     mask = (mask.cpu().numpy() * 255).astype(np.uint8)
#     Image.fromarray(mask).save(os.path.join(path, str(j) + '.png'))
#
# path = '/data/sets/czy/2'
# os.makedirs(path, exist_ok=True)
# for j, mask in enumerate(pts_msk):
#     mask = (mask.cpu().numpy() * 255).astype(np.uint8)
#     Image.fromarray(mask).save(os.path.join(path, str(j) + '.png'))
#
# path = '/data/sets/czy/3'
# os.makedirs(path, exist_ok=True)
# for j, mask in enumerate(pts_msk):
#     mask = (mask.cpu().numpy() * 255).astype(np.uint8)
#     Image.fromarray(mask).save(os.path.join(path, str(j) + '.png'))
#
# reg_min = row_reg.min(dim=2, keepdim=True)[0]
# reg_max = row_reg.max(dim=2, keepdim=True)[0]
# reg_msk = (row_reg - reg_min) / (reg_max - reg_min).clamp(min=1.0)
# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# print(reg_min, reg_max)
# path = '/data/sets/czy/4'
# os.makedirs(path, exist_ok=True)
# for j, mask in enumerate(reg_msk):
#     mask = (mask.cpu().numpy() * 255).astype(np.uint8)
#     Image.fromarray(mask).save(os.path.join(path, str(j) + '.png'))
#
# reg_min = col_reg.min(dim=1, keepdim=True)[0]
# reg_max = col_reg.max(dim=1, keepdim=True)[0]
# reg_msk = (col_reg - reg_min) / (reg_max - reg_min).clamp(min=1.0)
# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# print(reg_min, reg_max)
# path = '/data/sets/czy/5'
# os.makedirs(path, exist_ok=True)
# for j, mask in enumerate(reg_msk):
#     mask = (mask.cpu().numpy() * 255).astype(np.uint8)
#     Image.fromarray(mask).save(os.path.join(path, str(j) + '.png'))
