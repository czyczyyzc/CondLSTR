import numpy as np
import torch
import torch.nn.functional as F


class CondLSTR2DPost(object):

    def __init__(self,
                 min_points=5,
                 line_width=13,
                 score_thresh=0.3,
                 mask_downscale=4,
                 with_smooth=False):
        super(CondLSTR2DPost, self).__init__()
        self.min_points = min_points
        self.line_width = line_width
        self.score_thresh = score_thresh
        self.mask_downscale = mask_downscale
        self.with_smooth = with_smooth

    def __call__(self, logits_obj, logits_cls, regs, masks, lane_ranges, img_metas):
        device = logits_obj.device
        mask_h, mask_w = masks.shape[-2:]

        scores_obj = logits_obj.softmax(-1)[:, :, 0]                                                 # (N, L)
        scores_cls = logits_cls.softmax(-1)                                                          # (N, L, C)

        row_msks = masks.softmax(dim=3)                                                              # (N, L, H, W)
        row_locs = torch.arange(0, row_msks.size(3), dtype=torch.float32, device=device)             # (W,)
        row_locs = (row_locs * row_msks).sum(dim=3)                                                  # (N, L, H)

        row_locs = row_locs.round().long()
        row_regs = regs.gather(dim=3, index=row_locs.unsqueeze(3)).squeeze(3)                        # (N, L, H)
        row_locs = row_locs.float() + row_regs                                                       # (N, L, H)

        row_rngs = lane_ranges                                                                       # (N, L, 2)

        lane_points = []
        lane_attris = []
        lane_scores = []
        for row_loc, row_rng, scr_obj, scr_cls, img_meta in \
                zip(row_locs, row_rngs, scores_obj, scores_cls, img_metas):
            img_h, img_w = img_meta['img_shape'][:2]
            inp_h, inp_w = img_meta['input_shape'][:2]
            img_h, img_w = int(img_h * mask_h / inp_h), int(img_w * mask_w / inp_w)

            lan_kep = scr_obj >= self.score_thresh                                                   # (L,)
            row_loc = row_loc[lan_kep]                                                               # (M, H)
            row_rng = row_rng[lan_kep]                                                               # (M, 2)
            scr_cls = scr_cls[lan_kep]                                                               # (M, C)
            scr_obj = scr_obj[lan_kep]                                                               # (M,)

            row_sta = (row_rng[:, 0] * img_h).round().clamp(min=0, max=img_h-1).long()               # (M,)
            row_end = (row_rng[:, 1] * img_h).round().clamp(min=0, max=img_h-1).long()               # (M,)
            row_rng = torch.stack([row_sta, row_end], dim=1)                                         # (M, 2)

            lan_kep = (row_end - row_sta + 1) >= self.min_points
            row_loc = row_loc[lan_kep]                                                               # (M, H)
            row_rng = row_rng[lan_kep]                                                               # (M, 2)
            scr_cls = scr_cls[lan_kep]                                                               # (M, C)
            scr_obj = scr_obj[lan_kep]                                                               # (M,)
            
            attris = scr_cls.argmax(dim=1)                                                           # (M,)
            scores = scr_obj                                                                         # (M,)

            num_lanes = row_loc.size(0)

            points_list = []
            attris_list = []
            scores_list = []
            for lane_idx in range(num_lanes):
                selected_ys = torch.arange(row_rng[lane_idx, 0].item(),
                                           row_rng[lane_idx, 1].item() + 1, device=device)           # (P,)
                selected_xs = row_loc[lane_idx, selected_ys]                                         # (P,)
                selected_ys = selected_ys.float()                                                    # (P,)

                points = torch.stack([selected_xs, selected_ys], dim=1)                              # (P, 2)
                points = points * self.mask_downscale * 1.0

                if self.with_smooth:
                    point_ = points.cpu().numpy()
                    params = np.polyfit(point_[:, 1], point_[:, 0], 3)
                    p_func = np.poly1d(params)
                    pred_x = p_func(point_[:, 1])
                    diff_x = np.abs(pred_x - point_[:, 0])
                    keep_p = diff_x < 20  # self.line_width * self.mask_downscale / 2

                    if keep_p.shape[0] - keep_p.sum() > keep_p.shape[0] // 3:
                        continue
                    points = points[keep_p]

                    kH = 7
                    points = points.t()[None]                                                        # (1, 2, P)
                    points = F.pad(points, pad=((kH-1)//2, (kH-1)//2), mode='replicate')             # (1, 2, P)
                    points = F.avg_pool1d(points, kernel_size=kH, stride=1, padding=0)               # (1, 2, P)
                    points = points[0].t()

                points = points.cpu().numpy()
                attri = int(attris[lane_idx].cpu().numpy())
                score = float(scores[lane_idx].cpu().numpy())

                points_list.append(points)
                attris_list.append(attri)
                scores_list.append(score)

            lane_points.append(points_list)
            lane_attris.append(attris_list)
            lane_scores.append(scores_list)
        return lane_points, lane_attris, lane_scores
