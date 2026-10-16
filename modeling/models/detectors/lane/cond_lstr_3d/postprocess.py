import numpy as np
import torch
import torch.nn.functional as F


class CondLSTR3DPost(object):

    def __init__(self,
                 min_points=5,
                 line_width=13,
                 score_thresh=0.3,
                 mask_downscale=4,
                 with_smooth=False,
                 with_2d=True):
        super(CondLSTR3DPost, self).__init__()
        self.min_points = min_points
        self.line_width = line_width
        self.score_thresh = score_thresh
        self.mask_downscale = mask_downscale
        self.with_smooth = with_smooth
        self.with_2d = with_2d

    def __call__(self, logits_obj, logits_cls, masks_bev, masks_img, regs_bev, regs_img,
                 ranges_bev, ranges_img, img_metas):
        device = logits_obj.device

        scores_obj = logits_obj.softmax(-1)[:, :, 0]                                                   # (N, L)
        scores_cls = logits_cls.softmax(-1)                                                            # (N, L, C)

        row_msks_bev = masks_bev.softmax(dim=3)                                                        # (N, L, H, W)
        row_locs_bev = torch.arange(0, row_msks_bev.size(3), dtype=torch.float32, device=device)       # (W,)
        row_locs_bev = (row_locs_bev * row_msks_bev).sum(dim=3)                                        # (N, L, H)

        row_rgxs_bev = regs_bev[:, 0::2]                                                               # (N, L, H, W)
        row_rgzs_bev = regs_bev[:, 1::2]                                                               # (N, L, H, W)

        row_locs_bev = row_locs_bev.round().long()                                                     # (N, L, H)
        row_rgxs_bev = row_rgxs_bev.gather(dim=3, index=row_locs_bev.unsqueeze(3)).squeeze(3)          # (N, L, H)
        row_locs_bev = row_locs_bev.float() + row_rgxs_bev                                             # (N, L, H)

        row_ptxs_bev = row_locs_bev.round().long().clamp(min=0, max=masks_bev.size(3)-1)               # (N, L, H)
        row_rgzs_bev = row_rgzs_bev.gather(dim=3, index=row_ptxs_bev.unsqueeze(3)).squeeze(3)          # (N, L, H)
        row_rngs_bev = ranges_bev                                                                      # (N, L, 2)
        msk_size_bev = masks_bev.shape[-2:]                                                            # (H, W)

        if self.with_2d:
            row_msks_img = masks_img.softmax(dim=3)                                                    # (N, L, H, W)
            row_locs_img = torch.arange(0, row_msks_img.size(3), dtype=torch.float32, device=device)   # (W,)
            row_locs_img = (row_locs_img * row_msks_img).sum(dim=3)                                    # (N, L, H)

            row_rgxs_img = regs_img[:, 0::2]                                                           # (N, L, H, W)
            row_rgzs_img = regs_img[:, 1::2]                                                           # (N, L, H, W)

            row_locs_img = row_locs_img.round().long()                                                 # (N, L, H)
            row_rgxs_img = row_rgxs_img.gather(dim=3, index=row_locs_img.unsqueeze(3)).squeeze(3)      # (N, L, H)
            row_locs_img = row_locs_img.float() + row_rgxs_img                                         # (N, L, H)

            row_ptxs_img = row_locs_img.round().long().clamp(min=0, max=masks_img.size(3)-1)           # (N, L, H)
            row_rgzs_img = row_rgzs_img.gather(dim=3, index=row_ptxs_img.unsqueeze(3)).squeeze(3)      # (N, L, H)
            row_rngs_img = ranges_img                                                                  # (N, L, 2)
            msk_size_img = masks_img.shape[-2:]                                                      # (H, W)
        else:
            row_locs_img = [None for _ in range(len(masks_bev))]
            row_rgzs_img = [None for _ in range(len(masks_bev))]
            row_rngs_img = [None for _ in range(len(masks_bev))]
            msk_size_img = None

        lane_points = []
        lane_attris = []
        lane_scores = []
        for i, (row_loc_bev, row_loc_img, row_rng_bev, row_rng_img, row_rgz_bev, row_rgz_img, score_obj, score_cls, img_meta) in \
                enumerate(zip(row_locs_bev, row_locs_img, row_rngs_bev, row_rngs_img, row_rgzs_bev, row_rgzs_img, scores_obj, scores_cls, img_metas)):
            lane_keep = score_obj >= self.score_thresh                                                 # (L,)
            score_obj = score_obj[lane_keep]                                                           # (M,)
            score_cls = score_cls[lane_keep]                                                           # (M, C)

            row_loc_bev = row_loc_bev[lane_keep]                                                       # (M, H)
            row_rng_bev = row_rng_bev[lane_keep]                                                       # (M, 2)
            row_rgz_bev = row_rgz_bev[lane_keep]                                                       # (M, H)

            if self.with_2d:
                row_loc_img = row_loc_img[lane_keep]                                                   # (M, H)
                row_rng_img = row_rng_img[lane_keep]                                                   # (M, 2)
                row_rgz_img = row_rgz_img[lane_keep]                                                   # (M, H)

            bev_h, bev_w = msk_size_bev[-2:]
            row_sta_bev = (row_rng_bev[:, 0] * bev_h).round().clamp(min=0, max=bev_h-1).long()         # (M,)
            row_end_bev = (row_rng_bev[:, 1] * bev_h).round().clamp(min=0, max=bev_h-1).long()         # (M,)
            row_rng_bev = torch.stack([row_sta_bev, row_end_bev], dim=1)                               # (M, 2)
            lane_keep = (row_end_bev - row_sta_bev + 1) >= self.min_points                             # (M,)

            if self.with_2d:
                img_h, img_w = img_meta['img_shape'][:2]
                inp_h, inp_w = img_meta['input_shape'][:2]
                img_h, img_w = int(img_h * msk_size_img[0] / inp_h), int(img_w * msk_size_img[1] / inp_w)

                row_sta_img = (row_rng_img[:, 0] * img_h).round().clamp(min=0, max=img_h-1).long()     # (M,)
                row_end_img = (row_rng_img[:, 1] * img_h).round().clamp(min=0, max=img_h-1).long()     # (M,)
                row_rng_img = torch.stack([row_sta_img, row_end_img], dim=1)                           # (M, 2)
                lane_keep = lane_keep & ((row_end_img - row_sta_img + 1) >= self.min_points)           # (M,)

            score_obj = score_obj[lane_keep]                                                           # (M,)
            score_cls = score_cls[lane_keep]                                                           # (M, C)

            row_loc_bev = row_loc_bev[lane_keep]                                                       # (M, H)
            row_rng_bev = row_rng_bev[lane_keep]                                                       # (M, 2)
            row_rgz_bev = row_rgz_bev[lane_keep]                                                       # (M, H)

            if self.with_2d:
                row_loc_img = row_loc_img[lane_keep]                                                   # (M, H)
                row_rng_img = row_rng_img[lane_keep]                                                   # (M, 2)
                row_rgz_img = row_rgz_img[lane_keep]                                                   # (M, H)

            attris = score_cls.argmax(dim=1)                                                           # (M,)
            scores = score_obj                                                                         # (M,)
            num_lanes = score_obj.size(0)

            points_list = []
            attris_list = []
            scores_list = []
            for lane_idx in range(num_lanes):
                selected_ys = torch.arange(row_rng_bev[lane_idx, 0].item(),
                                           row_rng_bev[lane_idx, 1].item() + 1, device=device)         # (P,)
                selected_xs = row_loc_bev[lane_idx, selected_ys]                                       # (P,)
                selected_zs = row_rgz_bev[lane_idx, selected_ys]                                       # (P,)
                selected_ys = selected_ys.float()                                                      # (P,)

                selected_xs = selected_xs * self.mask_downscale * 1.0
                selected_ys = selected_ys * self.mask_downscale * 1.0
                points_bev = torch.stack([selected_xs, selected_ys, selected_zs], dim=1)               # (P, 3)
                points_bev = points_bev.cpu().numpy()                                                  # (P, 3)

                if self.with_2d:
                    selected_ys = torch.arange(row_rng_img[lane_idx, 0].item(),
                                               row_rng_img[lane_idx, 1].item() + 1, device=device)     # (P,)
                    selected_xs = row_loc_img[lane_idx, selected_ys]                                   # (P,)
                    selected_zs = row_rgz_img[lane_idx, selected_ys]                                   # (P,)
                    selected_ys = selected_ys.float()                                                  # (P,)

                    selected_xs = selected_xs * self.mask_downscale * 1.0
                    selected_ys = selected_ys * self.mask_downscale * 1.0
                    points_img = torch.stack([selected_xs, selected_ys, selected_zs], dim=1)           # (P, 3)
                    points_img = points_img.cpu().numpy()                                              # (P, 3)
                else:
                    points_img = None

                points = (points_bev, points_img)
                attri = int(attris[lane_idx].cpu().numpy())
                score = float(scores[lane_idx].cpu().numpy())
                points_list.append(points)
                attris_list.append(attri)
                scores_list.append(score)

            lane_points.append(points_list)
            lane_attris.append(attris_list)
            lane_scores.append(scores_list)
        return lane_points, lane_attris, lane_scores


# if self.with_smooth:
#     point_ = points.cpu().numpy()
#     params = np.polyfit(point_[:, 1], point_[:, 0], 3)
#     p_func = np.poly1d(params)
#     pred_x = p_func(point_[:, 1])
#     diff_x = np.abs(pred_x - point_[:, 0])
#     keep_p = diff_x < 20  # self.line_width * self.mask_downscale / 2
#     if keep_p.shape[0] - keep_p.sum() > keep_p.shape[0] // 3:
#         continue
#     points = points[keep_p]
#     kH = 7
#     points = points.t()[None]                                                        # (1, 2, P)
#     points = F.pad(points, pad=((kH-1)//2, (kH-1)//2), mode='replicate')             # (1, 2, P)
#     points = F.avg_pool1d(points, kernel_size=kH, stride=1, padding=0)               # (1, 2, P)
#     points = points[0].t()
