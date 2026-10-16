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
                 reg_weight=1., rng_weight=1., line_width=15, with_2d=True):
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
        self.with_2d    = with_2d

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
        loss_obj = F.cross_entropy(logits.transpose(1, 2), target_classes, self.empty_weight)  # (N, C, L) (N, L)
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

    def loss_loc(self, masks_bev, masks_img, targets, indices, num_lanes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The targets boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        device = masks_bev.device
        idx = self._get_src_permutation_idx(indices)

        # masks: (N, L, H, W)
        row_msks_bev = masks_bev.softmax(dim=3)                                                                    # (N, L, H, W)
        row_locs_bev = torch.arange(0, row_msks_bev.size(3), dtype=torch.float32, device=device)                   # (W,)
        row_locs_bev = (row_locs_bev * row_msks_bev).sum(dim=3)                                                    # (N, L, H)

        src_row_locs_bev = row_locs_bev[idx]                                                                       # (M1 + M2, H)
        tgt_row_locs_bev = torch.cat([tgt['gt_row_loc_bev'][i] for tgt, (_, i) in zip(targets, indices)])          # (M1 + M2, H)
        tgt_loc_msks_bev = torch.cat([tgt['gt_loc_msk_bev'][i] for tgt, (_, i) in zip(targets, indices)])          # (M1 + M2, H)

        loss_row_loc_bev = F.l1_loss(src_row_locs_bev, tgt_row_locs_bev, reduction='none')                         # (M1 + M2, H)
        loss_row_loc_bev = loss_row_loc_bev * tgt_loc_msks_bev                                                     # (M1 + M2, H)
        loss_row_loc_bev = loss_row_loc_bev.sum(dim=1) / tgt_loc_msks_bev.sum(dim=1).clamp(min=1.0)                # (M1 + M2,)
        loss_row_loc_bev = loss_row_loc_bev.sum() / num_lanes

        loss_row_loc = loss_row_loc_bev

        src_row_loc1_bev = src_row_locs_bev - self.line_width                                                      # (M1 + M2, H)
        src_row_loc2_bev = src_row_locs_bev + self.line_width                                                      # (M1 + M2, H)
        tgt_row_loc1_bev = tgt_row_locs_bev - self.line_width                                                      # (M1 + M2, H)
        tgt_row_loc2_bev = tgt_row_locs_bev + self.line_width                                                      # (M1 + M2, H)

        line_row_ovr_bev = (torch.min(src_row_loc2_bev, tgt_row_loc2_bev) -
                            torch.max(src_row_loc1_bev, tgt_row_loc1_bev)) * tgt_loc_msks_bev                      # (M1 + M2, H)
        line_row_uni_bev = (torch.max(src_row_loc2_bev, tgt_row_loc2_bev) -
                            torch.min(src_row_loc1_bev, tgt_row_loc1_bev)) * tgt_loc_msks_bev                      # (M1 + M2, H)
        line_row_iou_bev = line_row_ovr_bev.sum(dim=1) / (line_row_uni_bev.sum(dim=1) + 1e-9)                      # (M1 + M2)
        loss_row_iou_bev = 1 - line_row_iou_bev                                                                    # (M1 + M2)
        loss_row_iou_bev = loss_row_iou_bev.sum() / num_lanes

        loss_row_iou = loss_row_iou_bev

        if self.with_2d:
            row_msks_img = masks_img.softmax(dim=3)                                                                # (N, L, H, W)
            row_locs_img = torch.arange(0, row_msks_img.size(3), dtype=torch.float32, device=device)               # (W,)
            row_locs_img = (row_locs_img * row_msks_img).sum(dim=3)                                                # (N, L, H)

            src_row_locs_img = row_locs_img[idx]                                                                   # (M1 + M2, H)
            tgt_row_locs_img = torch.cat([tgt['gt_row_loc_img'][i] for tgt, (_, i) in zip(targets, indices)])      # (M1 + M2, H)
            tgt_loc_msks_img = torch.cat([tgt['gt_loc_msk_img'][i] for tgt, (_, i) in zip(targets, indices)])      # (M1 + M2, H)

            loss_row_loc_img = F.l1_loss(src_row_locs_img, tgt_row_locs_img, reduction='none')                     # (M1 + M2, H)
            loss_row_loc_img = loss_row_loc_img * tgt_loc_msks_img                                                 # (M1 + M2, H)
            loss_row_loc_img = loss_row_loc_img.sum(dim=1) / tgt_loc_msks_img.sum(dim=1).clamp(min=1.0)            # (M1 + M2,)
            loss_row_loc_img = loss_row_loc_img.sum() / num_lanes

            loss_row_loc = loss_row_loc + loss_row_loc_img

            src_row_loc1_img = src_row_locs_img - self.line_width                                                  # (M1 + M2, H)
            src_row_loc2_img = src_row_locs_img + self.line_width                                                  # (M1 + M2, H)
            tgt_row_loc1_img = tgt_row_locs_img - self.line_width                                                  # (M1 + M2, H)
            tgt_row_loc2_img = tgt_row_locs_img + self.line_width                                                  # (M1 + M2, H)

            line_row_ovr_img = (torch.min(src_row_loc2_img, tgt_row_loc2_img) -
                                torch.max(src_row_loc1_img, tgt_row_loc1_img)) * tgt_loc_msks_img                  # (M1 + M2, H)
            line_row_uni_img = (torch.max(src_row_loc2_img, tgt_row_loc2_img) -
                                torch.min(src_row_loc1_img, tgt_row_loc1_img)) * tgt_loc_msks_img                  # (M1 + M2, H)
            line_row_iou_img = line_row_ovr_img.sum(dim=1) / (line_row_uni_img.sum(dim=1) + 1e-9)                  # (M1 + M2)
            loss_row_iou_img = 1 - line_row_iou_img                                                                # (M1 + M2)
            loss_row_iou_img = loss_row_iou_img.sum() / num_lanes

            loss_row_iou = loss_row_iou + loss_row_iou_img

        loss_loc = loss_row_loc + loss_row_iou * 2.0

        # src_row_msks = masks[idx]
        # loss_row_cls = F.cross_entropy(src_row_msks.permute(0, 2, 1), tgt_row_locs.long(), reduction='none')     # (M1 + M2, H)
        # loss_row_cls = loss_row_cls * tgt_row_msks                                                               # (M1 + M2, H)
        # loss_row_cls = loss_row_cls.sum(dim=1) / tgt_row_msks.sum(dim=1).clamp(min=1.0)                          # (M1 + M2)
        # loss_row_cls = loss_row_cls.sum() / num_lanes
        #
        # loss_loc = loss_row_loc + loss_row_cls
        return loss_loc

    def loss_reg(self, regs_bev, regs_img, targets, indices, num_lanes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The targets boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # regs: (N, L * 2, H, W)
        idx = self._get_src_permutation_idx(indices)
        row_rgxs_bev = regs_bev[:, 0::2]                                                                           # (N, L, H, W)
        row_rgzs_bev = regs_bev[:, 1::2]                                                                           # (N, L, H, W)

        src_row_rgxs_bev = row_rgxs_bev[idx]                                                                       # (M1 + M2, H, W)
        src_row_rgzs_bev = row_rgzs_bev[idx]                                                                       # (M1 + M2, H, W)
        tgt_row_rgxs_bev = torch.cat([tgt['gt_row_rgx_bev'][i] for tgt, (_, i) in zip(targets, indices)])          # (M1 + M2, H, W)
        tgt_row_rgzs_bev = torch.cat([tgt['gt_row_rgz_bev'][i] for tgt, (_, i) in zip(targets, indices)])          # (M1 + M2, H, W)
        tgt_reg_msks_bev = torch.cat([tgt['gt_reg_msk_bev'][i] for tgt, (_, i) in zip(targets, indices)])          # (M1 + M2, H, W)

        loss_row_rgx_bev = F.l1_loss(src_row_rgxs_bev, tgt_row_rgxs_bev, reduction='none')                         # (M1 + M2, H, W)
        loss_row_rgz_bev = F.l1_loss(src_row_rgzs_bev, tgt_row_rgzs_bev, reduction='none')                         # (M1 + M2, H, W)
        loss_row_reg_bev = (loss_row_rgx_bev + loss_row_rgz_bev) * tgt_reg_msks_bev                                # (M1 + M2, H, W)
        loss_row_reg_bev = loss_row_reg_bev.sum(dim=(1, 2)) / tgt_reg_msks_bev.sum(dim=(1, 2)).clamp(min=1.0)      # (M1 + M2,)
        loss_row_reg_bev = loss_row_reg_bev.sum() / num_lanes

        loss_row_reg = loss_row_reg_bev

        if self.with_2d:
            row_rgxs_img = regs_img[:, 0::2]                                                                       # (N, L, H, W)
            row_rgzs_img = regs_img[:, 1::2]                                                                       # (N, L, H, W)

            src_row_rgxs_img = row_rgxs_img[idx]                                                                   # (M1 + M2, H, W)
            src_row_rgzs_img = row_rgzs_img[idx]                                                                   # (M1 + M2, H, W)
            tgt_row_rgxs_img = torch.cat([tgt['gt_row_rgx_img'][i] for tgt, (_, i) in zip(targets, indices)])      # (M1 + M2, H, W)
            tgt_row_rgzs_img = torch.cat([tgt['gt_row_rgz_img'][i] for tgt, (_, i) in zip(targets, indices)])      # (M1 + M2, H, W)
            tgt_reg_msks_img = torch.cat([tgt['gt_reg_msk_img'][i] for tgt, (_, i) in zip(targets, indices)])      # (M1 + M2, H, W)

            loss_row_rgx_img = F.l1_loss(src_row_rgxs_img, tgt_row_rgxs_img, reduction='none')                     # (M1 + M2, H, W)
            loss_row_rgz_img = 0.0
            # loss_row_rgz_img = F.l1_loss(src_row_rgzs_img, tgt_row_rgzs_img, reduction='none')                     # (M1 + M2, H, W)
            loss_row_reg_img = (loss_row_rgx_img + loss_row_rgz_img) * tgt_reg_msks_img                            # (M1 + M2, H, W)
            loss_row_reg_img = loss_row_reg_img.sum(dim=(1, 2)) / tgt_reg_msks_img.sum(dim=(1, 2)).clamp(min=1.0)  # (M1 + M2,)
            loss_row_reg_img = loss_row_reg_img.sum() / num_lanes

            loss_row_reg = loss_row_reg + loss_row_reg_img

        loss_reg = loss_row_reg
        return loss_reg

    def loss_rng(self, ranges_bev, ranges_img, targets, indices, num_lanes):
        # lane_ranges: (N, L, 2)
        idx = self._get_src_permutation_idx(indices)

        src_row_rngs_bev = ranges_bev[idx]                                                                         # (M1 + M2, 2)
        tgt_row_rngs_bev = torch.cat([tgt['gt_row_rng_bev'][i] for tgt, (_, i) in zip(targets, indices)])          # (M1 + M2, 2)
        loss_row_rng_bev = F.l1_loss(src_row_rngs_bev, tgt_row_rngs_bev, reduction='sum')
        loss_row_rng_bev = loss_row_rng_bev / num_lanes

        loss_row_rng = loss_row_rng_bev

        if self.with_2d:
            src_row_rngs_img = ranges_img[idx]                                                                      # (M1 + M2, 2)
            tgt_row_rngs_img = torch.cat([tgt['gt_row_rng_img'][i] for tgt, (_, i) in zip(targets, indices)])       # (M1 + M2, 2)
            loss_row_rng_img = F.l1_loss(src_row_rngs_img, tgt_row_rngs_img, reduction='sum')
            loss_row_rng_img = loss_row_rng_img / num_lanes

            loss_row_rng = loss_row_rng + loss_row_rng_img

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

    def forward(self, logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img,
                ranges_bev, ranges_img, targets, num_lanes):
        """ This performs the loss computation.
        Parameters:
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img,
                               ranges_bev, ranges_img, targets)

        # Compute all the requested losses
        loss_obj = self.loss_obj(logits_obj, targets, indices)
        loss_cls = self.loss_cls(logits_cls, targets, indices, num_lanes)
        loss_loc = self.loss_loc(masks_bev, masks_img, targets, indices, num_lanes)
        loss_reg = self.loss_reg(regs_bev, regs_img, targets, indices, num_lanes)
        loss_rng = self.loss_rng(ranges_bev, ranges_img, targets, indices, num_lanes)

        loss_obj = loss_obj * self.obj_weight
        loss_cls = loss_cls * self.cls_weight
        loss_loc = loss_loc * self.loc_weight
        loss_reg = loss_reg * self.reg_weight
        loss_rng = loss_rng * self.rng_weight
        return loss_obj, loss_cls, loss_loc, loss_reg, loss_rng


class CondLSTR3DLoss(nn.Module):

    def __init__(self, min_points=10, line_width=15, weight_dict=None, eos_coef=1., with_2d=True):
        """
        Args:
            weights is a dict which sets the weight of the loss
            eg. {hm_weight: 1, kp_weight: 1, ins_weight: 1}
        """
        super(CondLSTR3DLoss, self).__init__()
        self.min_points = min_points
        self.line_width = line_width
        self.with_2d = with_2d

        if weight_dict is None:
            weight_dict = {'obj_weight': 1.0, 'cls_weight': 1.0, 'loc_weight': 1.0, 'reg_weight': 1.0, 'rng_weight': 5.0}

        matcher = HungarianMatcher(obj_weight=weight_dict['obj_weight'],
                                   cls_weight=weight_dict['cls_weight'],
                                   loc_weight=weight_dict['loc_weight'],
                                   reg_weight=weight_dict['reg_weight'],
                                   rng_weight=weight_dict['rng_weight'],
                                   line_width=line_width,
                                   with_2d=with_2d)

        self.criterion = SetCriterion(matcher=matcher,
                                      eos_coef=eos_coef,
                                      obj_weight=weight_dict['obj_weight'],
                                      cls_weight=weight_dict['cls_weight'],
                                      loc_weight=weight_dict['loc_weight'],
                                      reg_weight=weight_dict['reg_weight'],
                                      rng_weight=weight_dict['rng_weight'],
                                      line_width=line_width,
                                      with_2d=with_2d)

    @torch.no_grad()
    def preprocess(self, gt_masks_bev, gt_masks_img, gt_labels, mask_shape_bev, mask_shape_img, img_metas):
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
                bev_h, bev_w = mask_shape_bev[:2]
                gt_mask_bev = F.interpolate(gt_mask_bev.float(), size=mask_shape_bev, mode='nearest')             # (M, 2, H, W)
                row_crd_bev = torch.arange(0, gt_mask_bev.size(3), dtype=torch.float32, device=device)            # (W,)
                row_loc_bev = (gt_mask_bev[:, 0] * row_crd_bev).sum(dim=2) / gt_mask_bev[:, 0].sum(dim=2).clamp(min=1.0)  # (M, H)
                loc_msk_bev = gt_mask_bev[:, 0].max(dim=2)[0]                                                     # (M, H)

                row_rgx_bev = row_loc_bev[:, :, None] - row_crd_bev                                               # (M, H, W)
                reg_msk_bev = (row_rgx_bev >= -self.line_width * 4) & (row_rgx_bev <= self.line_width * 4) & loc_msk_bev[:, :, None].bool()
                row_rgx_bev[~reg_msk_bev] = 0.0                                                                   # (M, H, W)

                row_ptz_bev = (gt_mask_bev[:, 0] * gt_mask_bev[:, 1]).sum(dim=2) / gt_mask_bev[:, 0].sum(dim=2).clamp(min=1.0)  # (M, H)
                row_rgz_bev = row_ptz_bev[:, :, None].repeat(1, 1, gt_mask_bev.size(3))                           # (M, H, W)
                row_rgz_bev[~reg_msk_bev] = 0.0                                                                   # (M, H, W)

                reg_msk_bev = reg_msk_bev.float()                                                                 # (M, H, W)
                row_sta_bev = loc_msk_bev.argmax(dim=1)                                                           # (M,)
                row_end_bev = loc_msk_bev.flip(1).argmax(dim=1)                                                   # (M,)
                row_end_bev = loc_msk_bev.size(1) - 1 - row_end_bev                                               # (M,)
                row_rng_bev = torch.stack([row_sta_bev, row_end_bev], dim=1) / bev_h                              # (M, 2)

                label_obj   = torch.zeros(len(gt_mask_bev), dtype=torch.int64, device=device)                     # (M,)
                label_cls   = gt_label                                                                            # (M,)
            else:
                row_loc_bev = torch.zeros((0, mask_shape_bev[0]), dtype=torch.float32, device=device)
                loc_msk_bev = torch.zeros((0, mask_shape_bev[0]), dtype=torch.float32, device=device)
                row_rgx_bev = torch.zeros((0, *mask_shape_bev), dtype=torch.float32, device=device)
                row_rgz_bev = torch.zeros((0, *mask_shape_bev), dtype=torch.float32, device=device)
                reg_msk_bev = torch.zeros((0, *mask_shape_bev), dtype=torch.float32, device=device)
                row_rng_bev = torch.zeros((0, 2), dtype=torch.float32, device=device)
                label_obj   = torch.zeros((0,), dtype=torch.int64, device=device)
                label_cls   = torch.zeros((0,), dtype=torch.int64, device=device)

            target = dict(
                gt_row_loc_bev=row_loc_bev,
                gt_loc_msk_bev=loc_msk_bev,
                gt_row_rgx_bev=row_rgx_bev,
                gt_row_rgz_bev=row_rgz_bev,
                gt_reg_msk_bev=reg_msk_bev,
                gt_row_rng_bev=row_rng_bev,
                gt_label_obj=label_obj,
                gt_label_cls=label_cls,
            )
            if self.with_2d:
                gt_mask_img = gt_mask_img[gt_keep]                                                                # (M, 2, H, W)
                if len(gt_mask_img) > 0:
                    img_h, img_w = img_meta['img_shape'][:2]
                    inp_h, inp_w = img_meta['input_shape'][:2]
                    img_h, img_w = int(img_h * mask_shape_img[0] / inp_h), int(img_w * mask_shape_img[1] / inp_w)

                    gt_mask_img = F.interpolate(gt_mask_img.float(), size=mask_shape_img, mode='nearest')         # (M, 2, H, W)
                    row_crd_img = torch.arange(0, gt_mask_img.size(3), dtype=torch.float32, device=device)        # (W,)
                    row_loc_img = (gt_mask_img[:, 0] * row_crd_img).sum(dim=2) / gt_mask_img[:, 0].sum(dim=2).clamp(min=1.0)  # (M, H)
                    loc_msk_img = gt_mask_img[:, 0].max(dim=2)[0]                                                 # (M, H)

                    row_rgx_img = row_loc_img[:, :, None] - row_crd_img                                           # (M, H, W)
                    reg_msk_img = (row_rgx_img >= -self.line_width * 4) & (row_rgx_img <= self.line_width * 4) & loc_msk_img[:, :, None].bool()
                    row_rgx_img[~reg_msk_img] = 0.0                                                               # (M, H, W)

                    row_ptz_img = (gt_mask_img[:, 0] * gt_mask_img[:, 1]).sum(dim=2) / gt_mask_img[:, 0].sum(dim=2).clamp(min=1.0)  # (M, H)
                    row_rgz_img = row_ptz_img[:, :, None].repeat(1, 1, gt_mask_img.size(3))                       # (M, H, W)
                    row_rgz_img[~reg_msk_img] = 0.0                                                               # (M, H, W)

                    reg_msk_img = reg_msk_img.float()                                                             # (M, H, W)
                    row_sta_img = loc_msk_img.argmax(dim=1)                                                       # (M,)
                    row_end_img = loc_msk_img.flip(1).argmax(dim=1)                                               # (M,)
                    row_end_img = loc_msk_img.size(1) - 1 - row_end_img                                           # (M,)
                    row_rng_img = torch.stack([row_sta_img, row_end_img], dim=1) / img_h                          # (M, 2)
                else:
                    row_loc_img = torch.zeros((0, mask_shape_img[0]), dtype=torch.float32, device=device)
                    loc_msk_img = torch.zeros((0, mask_shape_img[0]), dtype=torch.float32, device=device)
                    row_rgx_img = torch.zeros((0, *mask_shape_img), dtype=torch.float32, device=device)
                    row_rgz_img = torch.zeros((0, *mask_shape_img), dtype=torch.float32, device=device)
                    reg_msk_img = torch.zeros((0, *mask_shape_img), dtype=torch.float32, device=device)
                    row_rng_img = torch.zeros((0, 2), dtype=torch.float32, device=device)

                target.update(
                    gt_row_loc_img=row_loc_img,
                    gt_loc_msk_img=loc_msk_img,
                    gt_row_rgx_img=row_rgx_img,
                    gt_row_rgz_img=row_rgz_img,
                    gt_reg_msk_img=reg_msk_img,
                    gt_row_rng_img=row_rng_img,
                )
            if 'class_weight' in img_meta:
                target.update(class_weight=img_meta['class_weight'])
            targets.append(target)
        return targets

    def forward(self, logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img, ranges_bev, ranges_img,
                gt_masks_bev, gt_masks_img, gt_labels, img_metas):
        if isinstance(logits_obj, (list, tuple)):
            mask_shape_bev = (masks_bev[0].size(2), masks_bev[1].size(3))
            mask_shape_img = (masks_img[0].size(2), masks_img[1].size(3)) if self.with_2d else None
            device = logits_obj[0].device
        else:
            mask_shape_bev = (masks_bev.size(2), masks_bev.size(3))
            mask_shape_img = (masks_img.size(2), masks_img.size(3)) if self.with_2d else None
            device = logits_obj.device

        targets = self.preprocess(gt_masks_bev, gt_masks_img, gt_labels, mask_shape_bev, mask_shape_img, img_metas)

        # Compute the average number of targets boxes accross all nodes, for normalization purposes
        num_lanes = sum(tgt['gt_label_obj'].shape[0] for tgt in targets)
        num_lanes = torch.as_tensor([num_lanes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            dist.all_reduce(num_lanes)
        num_lanes = torch.clamp(num_lanes / get_world_size(), min=1).item()

        if isinstance(logits_obj, (list, tuple)):
            loss_obj, loss_cls, loss_loc, loss_reg, loss_rng = 0, 0, 0, 0, 0
            for logits_obj_, logits_cls_, masks_bev_, masks_img_, regs_bev_, regs_img_, ranges_bev_, ranges_img_ in \
                    zip(logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img, ranges_bev, ranges_img):
                loss_obj_, loss_cls_, loss_loc_, loss_reg_, loss_rng_ = self.criterion(
                    logits_obj_, logits_cls_, masks_bev_, masks_img_, regs_bev_, regs_img_,
                    ranges_bev_, ranges_img_, targets, num_lanes)
                loss_obj += loss_obj_
                loss_cls += loss_cls_
                loss_loc += loss_loc_
                loss_reg += loss_reg_
                loss_rng += loss_rng_
        else:
            loss_obj, loss_cls, loss_loc, loss_reg, loss_rng = self.criterion(
                logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img,
                ranges_bev, ranges_img, targets, num_lanes)
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
