import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .matcher import HungarianMatcher
from modeling.models.losses import BinaryDiceLoss
from modeling.models.utils import is_dist_avail_and_initialized, get_world_size, accuracy


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, eos_coef=1., obj_weight=1., cls_weight=1., msk_weight=1.,
                 reg_weight=1., line_width=15, with_2d=True):
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
        self.msk_weight = msk_weight
        self.reg_weight = reg_weight
        self.line_width = line_width
        self.with_2d    = with_2d

        self.dice_loss = BinaryDiceLoss(smooth=1, p=2, reduction='none')

        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_obj(self, logits, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        device = logits.device
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([tgt['gt_lbl_obj'][J].long() for tgt, (_, J) in zip(targets, indices)])  # (M1 + M2,)
        target_classes = torch.full(logits.shape[:2], 1, dtype=torch.int64, device=device)                      # (N, L)
        target_classes[idx] = target_classes_o                                                                  # (M1 + M2,)
        loss_obj = F.cross_entropy(logits.transpose(1, 2), target_classes, self.empty_weight)  # (N, C, L) (N, L)
        return loss_obj

    def loss_cls(self, logits, targets, indices, num_lanes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # logits: (N, L, 20)
        idx = self._get_src_permutation_idx(indices)
        src_logits = logits[idx]                                                                   # (M1 + M2, 20)
        tgt_labels = torch.cat([tgt['gt_lbl_cls'][i] for tgt, (_, i) in zip(targets, indices)])  # (M1 + M2,)
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
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)                                             # (N,)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return card_err

    def loss_msk(self, msks_bev, msks_img, targets, indices, num_lanes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The targets boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)

        # masks: (N, L, H, W)
        src_msks_bev = msks_bev[idx]                                                                               # (M1 + M2, H, W)
        tgt_msks_bev = torch.cat([tgt['gt_msk_bev'][i] for tgt, (_, i) in zip(targets, indices)])                  # (M1 + M2, H, W)

        # bce loss
        loss_bce_bev = F.binary_cross_entropy_with_logits(src_msks_bev, tgt_msks_bev, reduction='none')            # (M1 + M2, H, W)
        loss_bce_bev = loss_bce_bev.mean(dim=(1, 2))                                                               # (M1 + M2,)
        loss_bce_bev = loss_bce_bev.sum() / num_lanes

        # dice loss
        loss_iou_bev = self.dice_loss(src_msks_bev.sigmoid(), tgt_msks_bev)                                        # (M1 + M2,)
        loss_iou_bev = loss_iou_bev.sum() / num_lanes

        loss_msk_bev = loss_bce_bev + loss_iou_bev * 2.0

        if self.with_2d:
            src_msks_img = msks_img[idx]                                                                           # (M1 + M2, H, W)
            tgt_msks_img = torch.cat([tgt['gt_msk_img'][i] for tgt, (_, i) in zip(targets, indices)])              # (M1 + M2, H, W)

            # bce loss
            loss_bce_img = F.binary_cross_entropy_with_logits(src_msks_img, tgt_msks_img, reduction='none')        # (M1 + M2, H, W)
            loss_bce_img = loss_bce_img.mean(dim=(1, 2))                                                           # (M1 + M2,)
            loss_bce_img = loss_bce_img.sum() / num_lanes

            # dice loss
            loss_iou_img = self.dice_loss(src_msks_img.sigmoid(), tgt_msks_img)                                    # (M1 + M2,)
            loss_iou_img = loss_iou_img.sum() / num_lanes

            loss_msk_img = loss_bce_img + loss_iou_img * 2.0
        else:
            loss_msk_img = 0.0

        loss_msk = loss_msk_bev + loss_msk_img
        return loss_msk

    def loss_reg(self, regs_bev, regs_img, targets, indices, num_lanes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The targets boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # regs: (N, L * 2, H, W)
        idx = self._get_src_permutation_idx(indices)

        src_regs_bev = regs_bev[idx]                                                                               # (M1 + M2, 3, H, W)
        tgt_regs_bev = torch.cat([tgt['gt_reg_bev'][i] for tgt, (_, i) in zip(targets, indices)])                  # (M1 + M2, 3, H, W)
        tgt_msks_bev = torch.cat([tgt['gt_msk_bev'][i] for tgt, (_, i) in zip(targets, indices)])                  # (M1 + M2, H, W)

        loss_reg_bev = F.l1_loss(src_regs_bev, tgt_regs_bev, reduction='none')                                     # (M1 + M2, 3, H, W)
        loss_reg_bev = loss_reg_bev.sum(dim=1)                                                                     # (M1 + M2, H, W)
        loss_reg_bev = loss_reg_bev * tgt_msks_bev                                                                 # (M1 + M2, H, W)
        loss_reg_bev = loss_reg_bev.sum(dim=(1, 2)) / tgt_msks_bev.sum(dim=(1, 2)).clamp(min=1.0)                  # (M1 + M2,)
        loss_reg_bev = loss_reg_bev.sum() / num_lanes

        if self.with_2d:
            src_regs_img = regs_img[idx]                                                                           # (M1 + M2, 3, H, W)
            tgt_regs_img = torch.cat([tgt['gt_reg_img'][i] for tgt, (_, i) in zip(targets, indices)])              # (M1 + M2, 3, H, W)
            tgt_msks_img = torch.cat([tgt['gt_msk_img'][i] for tgt, (_, i) in zip(targets, indices)])              # (M1 + M2, H, W)

            # loss_reg_img = F.l1_loss(src_regs_img, tgt_regs_img, reduction='none')                                 # (M1 + M2, 3, H, W)
            loss_reg_img = F.l1_loss(src_regs_img[:, :2], tgt_regs_img[:, :2], reduction='none')                   # (M1 + M2, 2, H, W)
            loss_reg_img = loss_reg_img.sum(dim=1)                                                                 # (M1 + M2, H, W)
            loss_reg_img = loss_reg_img * tgt_msks_img                                                             # (M1 + M2, H, W)
            loss_reg_img = loss_reg_img.sum(dim=(1, 2)) / tgt_msks_img.sum(dim=(1, 2)).clamp(min=1.0)              # (M1 + M2,)
            loss_reg_img = loss_reg_img.sum() / num_lanes
        else:
            loss_reg_img = 0.0

        loss_reg = loss_reg_bev + loss_reg_img
        return loss_reg

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])                           # (N * L,)
        src_idx = torch.cat([src for (src, _) in indices])                                                               # (N * L,)
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])                           # (N * L,)
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])                                                               # (N * L,)
        return batch_idx, tgt_idx

    def forward(self, logits_obj, logits_cls, msks_bev, regs_bev, msks_img, regs_img, targets, num_lanes):
        """ This performs the loss computation.
        Parameters:
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(logits_obj, logits_cls, msks_bev, regs_bev, msks_img, regs_img, targets)

        # Compute all the requested losses
        loss_obj = self.loss_obj(logits_obj, targets, indices)
        loss_cls = self.loss_cls(logits_cls, targets, indices, num_lanes)
        loss_msk = self.loss_msk(msks_bev, msks_img, targets, indices, num_lanes)
        loss_reg = self.loss_reg(regs_bev, regs_img, targets, indices, num_lanes)

        loss_obj = loss_obj * self.obj_weight
        loss_cls = loss_cls * self.cls_weight
        loss_msk = loss_msk * self.msk_weight
        loss_reg = loss_reg * self.reg_weight
        return loss_obj, loss_cls, loss_msk, loss_reg


class CondLSTRVote3DLoss(nn.Module):

    def __init__(self, min_points=10, line_width=15, weight_dict=None, eos_coef=1., with_2d=True):
        """
        Args:
            weights is a dict which sets the weight of the loss
            eg. {hm_weight: 1, kp_weight: 1, ins_weight: 1}
        """
        super(CondLSTRVote3DLoss, self).__init__()
        self.min_points = min_points
        self.line_width = line_width
        self.with_2d = with_2d

        if weight_dict is None:
            weight_dict = {'obj_weight': 1.0, 'cls_weight': 1.0, 'msk_weight': 1.0, 'reg_weight': 1.0}

        matcher = HungarianMatcher(obj_weight=weight_dict['obj_weight'],
                                   cls_weight=weight_dict['cls_weight'],
                                   msk_weight=weight_dict['msk_weight'],
                                   reg_weight=weight_dict['reg_weight'],
                                   line_width=line_width,
                                   with_2d=with_2d)

        self.criterion = SetCriterion(matcher=matcher,
                                      eos_coef=eos_coef,
                                      obj_weight=weight_dict['obj_weight'],
                                      cls_weight=weight_dict['cls_weight'],
                                      msk_weight=weight_dict['msk_weight'],
                                      reg_weight=weight_dict['reg_weight'],
                                      line_width=line_width,
                                      with_2d=with_2d)

    @torch.no_grad()
    def preprocess(self, gt_masks_bev, gt_masks_img, gt_labels, shape_bev, shape_img, img_metas):
        if gt_labels is None:
            gt_labels = [torch.zeros(len(gt_mask_bev), dtype=torch.int64, device=gt_masks_bev.device)
                         for gt_mask_bev in gt_masks_bev]
        if gt_masks_img is None:
            gt_masks_img = [None for _ in range(len(gt_masks_bev))]

        targets = []
        for gt_mask_bev, gt_mask_img, gt_label, img_meta in zip(gt_masks_bev, gt_masks_img, gt_labels, img_metas):
            device = gt_mask_bev.device

            assert len(gt_mask_bev) == len(gt_label), f"bev mask num: {len(gt_mask_bev)} vs label num: {len(gt_label)}"
            gt_keep = gt_mask_bev[:, 0].max(dim=2)[0].sum(dim=1) >= self.min_points                               # (M,)

            if self.with_2d:
                assert len(gt_mask_img) == len(gt_label), f"img mask num: {len(gt_mask_img)} vs label num: {len(gt_label)}"
                gt_keep = gt_keep & (gt_mask_img[:, 0].max(dim=2)[0].sum(dim=1) >= self.min_points)               # (M,)

            gt_label = gt_label[gt_keep]                                                                          # (M,)
            gt_mask_bev = gt_mask_bev[gt_keep]                                                                    # (M, 2, H, W)
            if len(gt_mask_bev) > 0:
                gt_mask = F.interpolate(gt_mask_bev.float(), size=shape_bev, mode='nearest')                      # (M, 2, H, W)
                pt_mask = F.max_pool2d(gt_mask[:, :1], kernel_size=self.line_width * 6 + 1,
                                       stride=1, padding=self.line_width * 3)[:, 0]                               # (M, H, W)
                msk_bev = torch.zeros_like(pt_mask, dtype=torch.float32)                                          # (M, H, W)
                rgx_bev = torch.zeros_like(pt_mask, dtype=torch.float32)                                          # (M, H, W)
                rgy_bev = torch.zeros_like(pt_mask, dtype=torch.float32)                                          # (M, H, W)
                rgz_bev = torch.zeros_like(pt_mask, dtype=torch.float32)                                          # (M, H, W)
                for i in range(len(gt_mask)):
                    src_idx = pt_mask[i].nonzero(as_tuple=False)                                                  # (P, 2)
                    tgt_idx = gt_mask[i, 0].nonzero(as_tuple=False)                                               # (G, 2)
                    tgt_rgz = gt_mask[i, 1, tgt_idx[:, 0], tgt_idx[:, 1]]                                         # (G,)
                    src_pts, tgt_pts = src_idx.float(), tgt_idx.float()
                    pts_dst, pts_idx = torch.cdist(src_pts, tgt_pts, p=2).min(dim=1)                              # (P,)
                    rgx_bev[i, src_idx[:, 0], src_idx[:, 1]] = tgt_pts[pts_idx, 1] - src_pts[:, 1]                # (M, H, W)
                    rgy_bev[i, src_idx[:, 0], src_idx[:, 1]] = tgt_pts[pts_idx, 0] - src_pts[:, 0]                # (M, H, W)
                    rgz_bev[i, src_idx[:, 0], src_idx[:, 1]] = tgt_rgz[pts_idx]                                   # (M, H, W)
                    msk_bev[i, src_idx[:, 0], src_idx[:, 1]] = torch.exp(-(pts_dst ** 2) / (2 * self.line_width ** 2))  # (M, H, W)
                reg_bev = torch.stack([rgx_bev, rgy_bev, rgz_bev], dim=1)                                         # (M, 3, H, W)
                lbl_obj = torch.zeros(len(gt_mask_bev), dtype=torch.int64, device=device)                         # (M,)
                lbl_cls = gt_label                                                                                # (M,)
            else:
                msk_bev = torch.zeros((0, *shape_bev), dtype=torch.float32, device=device)
                reg_bev = torch.zeros((0, 3, *shape_bev), dtype=torch.float32, device=device)
                lbl_obj = torch.zeros((0,), dtype=torch.int64, device=device)
                lbl_cls = torch.zeros((0,), dtype=torch.int64, device=device)

            target = dict(
                gt_msk_bev=msk_bev,
                gt_reg_bev=reg_bev,
                gt_lbl_obj=lbl_obj,
                gt_lbl_cls=lbl_cls,
            )
            if self.with_2d:
                gt_mask_img = gt_mask_img[gt_keep]                                                                # (M, 2, H, W)
                if len(gt_mask_img) > 0:
                    gt_mask = F.interpolate(gt_mask_img.float(), size=shape_img, mode='nearest')                  # (M, 2, H, W)
                    pt_mask = F.max_pool2d(gt_mask[:, :1], kernel_size=self.line_width * 6 + 1,
                                           stride=1, padding=self.line_width * 3)[:, 0]                           # (M, H, W)
                    msk_img = torch.zeros_like(pt_mask, dtype=torch.float32)                                      # (M, H, W)
                    rgx_img = torch.zeros_like(pt_mask, dtype=torch.float32)                                      # (M, H, W)
                    rgy_img = torch.zeros_like(pt_mask, dtype=torch.float32)                                      # (M, H, W)
                    rgz_img = torch.zeros_like(pt_mask, dtype=torch.float32)                                      # (M, H, W)
                    for i in range(len(gt_mask_img)):
                        src_idx = pt_mask[i].nonzero(as_tuple=False)                                              # (P, 2)
                        tgt_idx = gt_mask[i, 0].nonzero(as_tuple=False)                                           # (G, 2)
                        tgt_rgz = gt_mask[i, 1, tgt_idx[:, 0], tgt_idx[:, 1]]                                     # (G,)
                        src_pts, tgt_pts = src_idx.float(), tgt_idx.float()
                        pts_dst, pts_idx = torch.cdist(src_pts, tgt_pts, p=2).min(dim=1)                          # (P,)
                        rgx_img[i, src_idx[:, 0], src_idx[:, 1]] = tgt_pts[pts_idx, 1] - src_pts[:, 1]            # (M, H, W)
                        rgy_img[i, src_idx[:, 0], src_idx[:, 1]] = tgt_pts[pts_idx, 0] - src_pts[:, 0]            # (M, H, W)
                        rgz_img[i, src_idx[:, 0], src_idx[:, 1]] = tgt_rgz[pts_idx]                               # (M, H, W)
                        msk_img[i, src_idx[:, 0], src_idx[:, 1]] = torch.exp(-(pts_dst ** 2) / (2 * self.line_width ** 2))  # (M, H, W)
                    reg_img = torch.stack([rgx_img, rgy_img, rgz_img], dim=1)                                     # (M, 3, H, W)
                else:
                    msk_img = torch.zeros((0, *shape_img), dtype=torch.float32, device=device)
                    reg_img = torch.zeros((0, 3, *shape_img), dtype=torch.float32, device=device)

                target.update(
                    gt_msk_img=msk_img,
                    gt_reg_img=reg_img,
                )
            if 'class_weight' in img_meta:
                target.update(class_weight=img_meta['class_weight'])
            targets.append(target)
        return targets

    def forward(self, logits_obj, logits_cls, msks_bev, regs_bev, msks_img, regs_img,
                gt_masks_bev, gt_masks_img, gt_labels, img_metas):
        if isinstance(logits_obj, (list, tuple)):
            shape_bev = (msks_bev[0].size(2), msks_bev[0].size(3))
            shape_img = (msks_img[0].size(2), msks_img[0].size(3)) if self.with_2d else None
            device = logits_obj[0].device
        else:
            shape_bev = (msks_bev.size(2), msks_bev.size(3))
            shape_img = (msks_img.size(2), msks_img.size(3)) if self.with_2d else None
            device = logits_obj.device

        targets = self.preprocess(gt_masks_bev, gt_masks_img, gt_labels, shape_bev, shape_img, img_metas)

        # Compute the average number of targets boxes accross all nodes, for normalization purposes
        num_lanes = sum(tgt['gt_lbl_obj'].shape[0] for tgt in targets)
        num_lanes = torch.as_tensor([num_lanes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            dist.all_reduce(num_lanes)
        num_lanes = torch.clamp(num_lanes / get_world_size(), min=1).item()

        if isinstance(logits_obj, (list, tuple)):
            loss_obj, loss_cls, loss_msk, loss_reg = 0, 0, 0, 0
            for logits_obj_, logits_cls_, msks_bev_, regs_bev_, msks_img_, regs_img_ in \
                    zip(logits_obj, logits_cls, msks_bev, regs_bev, msks_img, regs_img):
                loss_obj_, loss_cls_, loss_msk_, loss_reg_ = self.criterion(
                    logits_obj_, logits_cls_, msks_bev_, regs_bev_, msks_img_, regs_img_, targets, num_lanes)
                loss_obj += loss_obj_
                loss_cls += loss_cls_
                loss_msk += loss_msk_
                loss_reg += loss_reg_
        else:
            loss_obj, loss_cls, loss_msk, loss_reg = self.criterion(
                logits_obj, logits_cls, msks_bev, regs_bev, msks_img, regs_img, targets, num_lanes)
        return loss_obj, loss_cls, loss_msk, loss_reg


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
