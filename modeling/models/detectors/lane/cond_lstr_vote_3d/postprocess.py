import numpy as np
import torch
import torch.nn.functional as F


class CondLSTRVote3DPost(object):

    def __init__(self,
                 min_points=5,
                 line_width=13,
                 score_thresh=0.3,
                 mask_downscale=4,
                 with_smooth=False,
                 with_2d=True):
        super(CondLSTRVote3DPost, self).__init__()
        self.min_points = min_points
        self.line_width = line_width
        self.score_thresh = score_thresh
        self.mask_downscale = mask_downscale
        self.with_smooth = with_smooth
        self.with_2d = with_2d

    def __call__(self, logits_obj, logits_cls, msks_bev, regs_bev, msks_img, regs_img, img_metas):
        device = logits_obj.device

        scrs_obj = logits_obj.softmax(-1)[:, :, 0]                                                     # (N, L)
        scrs_cls = logits_cls.softmax(-1)                                                              # (N, L, C)

        cdxs_bev = torch.arange(0, msks_bev.size(3), dtype=torch.float32, device=device)               # (W,)
        cdys_bev = torch.arange(0, msks_bev.size(2), dtype=torch.float32, device=device)               # (H,)
        ptxs_bev = regs_bev[:, :, 0] + cdxs_bev                                                        # (N, L, H, W)
        ptys_bev = regs_bev[:, :, 1] + cdys_bev[:, None]                                               # (N, L, H, W)
        ptzs_bev = regs_bev[:, :, 2]                                                                   # (N, L, H, W)

        ptxs_bev = ptxs_bev.round().clamp(min=0, max=msks_bev.size(3)-1).long()                        # (N, L, H, W)
        ptys_bev = ptys_bev.round().clamp(min=0, max=msks_bev.size(2)-1).long()                        # (N, L, H, W)
        pnts_bev = ptys_bev * msks_bev.size(3) + ptxs_bev                                              # (N, L, H, W)
        scrs_bev = torch.exp(-(regs_bev[:, :, 0] ** 2 + regs_bev[:, :, 1] ** 2) / (2 * self.line_width ** 2))  # (N, L, H, W)
        scrs_bev = (scrs_bev * msks_bev.sigmoid()).sqrt()                                              # (N, L, H, W)

        pnts_bev = pnts_bev.flatten(2)                                                                 # (N, L, H * W)
        scrs_bev = scrs_bev.flatten(2)                                                                 # (N, L, H * W)
        vots_bev = torch.zeros_like(scrs_bev)                                                          # (N, L, H * W)
        vots_bev.scatter_add_(2, pnts_bev, scrs_bev)                                                   # (N, L, H * W)
        vots_bev = vots_bev.view(msks_bev.shape)                                                       # (N, L, H, W)

        if self.with_2d:
            cdxs_img = torch.arange(0, msks_img.size(3), dtype=torch.float32, device=device)           # (W,)
            cdys_img = torch.arange(0, msks_img.size(2), dtype=torch.float32, device=device)           # (H,)
            ptxs_img = regs_img[:, :, 0] + cdxs_img                                                    # (N, L, H, W)
            ptys_img = regs_img[:, :, 1] + cdys_img[:, None]                                           # (N, L, H, W)
            ptzs_img = regs_img[:, :, 2]                                                               # (N, L, H, W)

            ptxs_img = ptxs_img.round().clamp(min=0, max=msks_img.size(3)-1).long()                    # (N, L, H, W)
            ptys_img = ptys_img.round().clamp(min=0, max=msks_img.size(2)-1).long()                    # (N, L, H, W)
            pnts_img = ptys_img * msks_img.size(3) + ptxs_img                                          # (N, L, H, W)
            scrs_img = torch.exp(-(regs_img[:, :, 0] ** 2 + regs_img[:, :, 1] ** 2) / (2 * self.line_width ** 2))  # (N, L, H, W)
            scrs_img = (scrs_img * msks_img.sigmoid()).sqrt()                                          # (N, L, H, W)

            pnts_img = pnts_img.flatten(2)                                                             # (N, L, H * W)
            scrs_img = scrs_img.flatten(2)                                                             # (N, L, H * W)
            vots_img = torch.zeros_like(scrs_img)                                                      # (N, L, H * W)
            vots_img.scatter_add_(2, pnts_img, scrs_img)                                               # (N, L, H * W)
            vots_img = vots_img.view(msks_img.shape)                                                   # (N, L, H, W)
        else:
            vots_img = [None for _ in range(scrs_obj.size(0))]
            ptzs_img = [None for _ in range(scrs_obj.size(0))]

        lane_points = []
        lane_attris = []
        lane_scores = []
        for i, (vot_bev, ptz_bev, vot_img, ptz_img, scr_obj, scr_cls, img_meta) in enumerate(zip(
            vots_bev, ptzs_bev, vots_img, ptzs_img, scrs_obj, scrs_cls, img_metas)):

            lan_kep = scr_obj >= self.score_thresh                                                     # (L,)
            scr_obj = scr_obj[lan_kep]                                                                 # (M,)
            scr_cls = scr_cls[lan_kep]                                                                 # (M, C)
            vot_bev = vot_bev[lan_kep]                                                                 # (M, H, W)
            ptz_bev = ptz_bev[lan_kep]                                                                 # (M, H, W)
            if self.with_2d:
                vot_img = vot_img[lan_kep]                                                             # (M, H, W)
                ptz_img = ptz_img[lan_kep]                                                             # (M, H, W)

            scr_cls, attris = scr_cls.max(dim=1)                                                       # (M,), (M,)
            scores = (scr_obj * scr_cls).sqrt()                                                        # (M,)
            num_lanes = scores.size(0)

            points_list = []
            attris_list = []
            scores_list = []
            for lane_idx in range(num_lanes):
                pnts_bev = (vot_bev[lane_idx] >= self.line_width).nonzero(as_tuple=False)              # (P, 2)
                ptzs_bev = ptz_bev[lane_idx, pnts_bev[:, 0], pnts_bev[:, 1]]                           # (P,)
                pnts_bev = pnts_bev.float() * self.mask_downscale                                      # (P, 2)
                pnts_bev = pnts_bev.flip(1)                                                            # (P, 2)
                pnts_bev = torch.cat([pnts_bev, ptzs_bev[:, None]], dim=1)                             # (P, 3)
                pnts_bev = pnts_bev.cpu().numpy()                                                      # (P, 3)
                if pnts_bev.shape[0] < self.min_points:
                    continue

                if self.with_2d:
                    pnts_img = (vot_img[lane_idx] >= self.line_width).nonzero(as_tuple=False)          # (P, 2)
                    ptzs_img = ptz_img[lane_idx, pnts_img[:, 0], pnts_img[:, 1]]                       # (P,)
                    pnts_img = pnts_img.float() * self.mask_downscale                                  # (P, 2)
                    pnts_img = pnts_img.flip(1)                                                        # (P, 2)
                    pnts_img = torch.cat([pnts_img, ptzs_img[:, None]], dim=1)                         # (P, 3)
                    pnts_img = pnts_img.cpu().numpy()                                                  # (P, 3)
                    if pnts_img.shape[0] < self.min_points:
                        continue
                else:
                    pnts_img = None

                points = (pnts_bev, pnts_img)
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
