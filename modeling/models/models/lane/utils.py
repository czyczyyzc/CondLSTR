import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionEmbedding(nn.Module):

    def __init__(self, num_pos_feats=64, temperature=10000, ego_range=[-10, 3, -5, 10, 103, 5]):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.ego_range = ego_range
        self.scale = 2 * math.pi

    def position_encoder(self, coords3d):
        device = coords3d.device
        B, _, H, W = coords3d.shape

        pos_x = coords3d[:, 0].clamp(min=self.ego_range[0], max=self.ego_range[3])                       # (B, H, W)
        pos_y = coords3d[:, 1].clamp(min=self.ego_range[1], max=self.ego_range[4])                       # (B, H, W)
        pos_z = coords3d[:, 2].clamp(min=self.ego_range[2], max=self.ego_range[5])                       # (B, H, W)

        pos_x = (pos_x - self.ego_range[0]) / (self.ego_range[3] - self.ego_range[0])                    # (B, H, W)
        pos_y = (pos_y - self.ego_range[1]) / (self.ego_range[4] - self.ego_range[1])                    # (B, H, W)
        pos_z = (pos_z - self.ego_range[2]) / (self.ego_range[5] - self.ego_range[2])                    # (B, H, W)

        dim_t = torch.arange(self.num_pos_feats // 3, dtype=torch.float32, device=device)                # (C,)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.num_pos_feats // 3))                       # (C,)
        pad_n = self.num_pos_feats - (self.num_pos_feats // 3) * 3

        pos_x = pos_x[:, None] * self.scale / dim_t[:, None, None]                                       # (B, C, H, W)
        pos_y = pos_y[:, None] * self.scale / dim_t[:, None, None]                                       # (B, C, H, W)
        pos_z = pos_z[:, None] * self.scale / dim_t[:, None, None]                                       # (B, C, H, W)

        pos_x = torch.cat((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=1)                           # (B, C, H, W)
        pos_y = torch.cat((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=1)                           # (B, C, H, W)
        pos_z = torch.cat((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=1)                           # (B, C, H, W)

        pos_p = torch.zeros((B, pad_n, H, W), dtype=torch.float32, device=device)                        # (B, 1, H, W)
        embed = torch.cat((pos_x, pos_y, pos_z, pos_p), dim=1)                                           # (B, C, H, W)
        return embed

    def forward(self, bev_feat, img_feat, bev_regz, img_regz, depth_prior, img_metas):
        B, C, H, W = img_feat.size()
        img_h, img_w = img_metas[0]['input_shape'][:2]
        bev_h, bev_w = img_metas[0]['bev_shape'][:2]
        device = img_feat.device

        if len(depth_prior.shape) < 4:
            depth_prior = depth_prior.unsqueeze(1)

        coords_d = F.interpolate(depth_prior, size=(H, W), mode='bilinear', align_corners=True)          # (B, 1, H, W)
        coords_d = coords_d + img_regz                                                                   # (B, 1, H, W)
        coords_v = torch.linspace(0, img_h, H, dtype=torch.float32, device=device)                       # (H,)
        coords_u = torch.linspace(0, img_w, W, dtype=torch.float32, device=device)                       # (W,)
        coords2d = torch.stack(torch.meshgrid(coords_u, coords_v, indexing='xy'), dim=0)                 # (2, H, W)
        coords2d = coords2d.unsqueeze(0).repeat(B, 1, 1, 1)                                              # (B, 2, H, W)
        coords3d = torch.cat((coords2d * coords_d, coords_d, torch.ones_like(coords_d)), dim=1)          # (B, 3, H, W)

        ego2imgs = []
        for img_meta in img_metas:
            ego2img = np.eye(4)                                                                          # (4, 4)
            ego2img[:3] = img_meta['calibration']['trans_ego2img']                                       # (3, 4)
            ego2imgs.append(ego2img)
        ego2imgs = np.array(ego2imgs, dtype=np.float32)                                                  # (B, 4, 4)
        ego2imgs = torch.from_numpy(ego2imgs).to(device)                                                 # (B, 4, 4)
        coords3d = torch.inverse(ego2imgs).bmm(coords3d.reshape(B, 4, -1))                               # (B, 4, H * W)
        coords3d = coords3d.reshape(B, 4, H, W)[:, :-1]                                                  # (B, 3, H, W)
        img_embd = self.position_encoder(coords3d)                                                       # (B, C, H, W)

        B, C, H, W = bev_feat.size()
        coords_z = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)                         # (B, 1, H, W)
        coords_z = coords_z + bev_regz                                                                   # (B, 1, H, W)
        coords_y = torch.linspace(0, bev_h, H, dtype=torch.float32, device=device)                       # (H,)
        coords_x = torch.linspace(0, bev_w, W, dtype=torch.float32, device=device)                       # (W,)
        coords2d = torch.stack(torch.meshgrid(coords_x, coords_y, indexing='xy'), dim=0)                 # (2, H, W)
        coords2d = coords2d.unsqueeze(0).repeat(B, 1, 1, 1)                                              # (B, 2, H, W)
        coords3d = torch.cat((coords2d, coords_z, torch.ones_like(coords_z)), dim=1)                     # (B, 4, H, W)

        bev2egos = []
        for img_meta in img_metas:
            bev2egos.append(img_meta['calibration']['trans_bev2ego'])
        bev2egos = np.array(bev2egos, dtype=np.float32)                                                  # (B, 4, 4)
        bev2egos = torch.from_numpy(bev2egos).to(device)                                                 # (B, 4, 4)
        coords3d = bev2egos.bmm(coords3d.reshape(B, 4, -1))                                              # (B, 4, H * W)
        coords3d = coords3d.reshape(B, 4, H, W)[:, :-1]                                                  # (B, 3, H, W)
        bev_embd = self.position_encoder(coords3d)                                                       # (B, C, H, W)
        return bev_embd, img_embd


# class PositionEmbedding(nn.Module):

#     def __init__(self, num_pos_feats=64, temperature=10000, ego_range=[-10, 3, -5, 10, 103, 5]):
#         super().__init__()
#         self.num_pos_feats = num_pos_feats
#         self.temperature = temperature
#         self.ego_range = ego_range
#         self.scale = 2 * math.pi

#     def position_encoder(self, coords3d):
#         device = coords3d.device
#         B, _, H, W = coords3d.shape

#         pos_x = coords3d[:, 0].clamp(min=self.ego_range[0], max=self.ego_range[3])                       # (B, H, W)
#         pos_y = coords3d[:, 1].clamp(min=self.ego_range[1], max=self.ego_range[4])                       # (B, H, W)
#         pos_z = coords3d[:, 2].clamp(min=self.ego_range[2], max=self.ego_range[5])                       # (B, H, W)

#         pos_x = (pos_x - self.ego_range[0]) / (self.ego_range[3] - self.ego_range[0])                    # (B, H, W)
#         pos_y = (pos_y - self.ego_range[1]) / (self.ego_range[4] - self.ego_range[1])                    # (B, H, W)
#         pos_z = (pos_z - self.ego_range[2]) / (self.ego_range[5] - self.ego_range[2])                    # (B, H, W)

#         dim_t = torch.arange(self.num_pos_feats // 3, dtype=torch.float32, device=device)                # (C,)
#         dim_t = self.temperature ** (2 * (dim_t // 2) / (self.num_pos_feats // 3))                       # (C,)
#         pad_n = self.num_pos_feats - (self.num_pos_feats // 3) * 3

#         pos_x = pos_x[:, None] * self.scale / dim_t[:, None, None]                                       # (B, C, H, W)
#         pos_y = pos_y[:, None] * self.scale / dim_t[:, None, None]                                       # (B, C, H, W)
#         pos_z = pos_z[:, None] * self.scale / dim_t[:, None, None]                                       # (B, C, H, W)

#         pos_x = torch.cat((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=1)                           # (B, C, H, W)
#         pos_y = torch.cat((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=1)                           # (B, C, H, W)
#         pos_z = torch.cat((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=1)                           # (B, C, H, W)

#         pos_p = torch.zeros((B, pad_n, H, W), dtype=torch.float32, device=device)                        # (B, 1, H, W)
#         embed = torch.cat((pos_x, pos_y, pos_z, pos_p), dim=1)                                           # (B, C, H, W)
#         return embed

#     def forward(self, bev_feat, img_feat, depth_prior, img_metas):
#         B, C, H, W = img_feat.size()
#         img_h, img_w = img_metas[0]['input_shape'][:2]
#         bev_h, bev_w = img_metas[0]['bev_shape'][:2]
#         device = img_feat.device

#         if len(depth_prior.shape) < 4:
#             depth_prior = depth_prior.unsqueeze(1)                                                       # (B, 1, H, W)

#         coords_d = F.interpolate(depth_prior, size=(H, W), mode='bilinear', align_corners=True)          # (B, 1, H, W)
#         coords_v = torch.linspace(0, img_h, H, dtype=torch.float32, device=device)                       # (H,)
#         coords_u = torch.linspace(0, img_w, W, dtype=torch.float32, device=device)                       # (W,)
#         coords2d = torch.stack(torch.meshgrid(coords_u, coords_v, indexing='xy'), dim=0)                 # (2, H, W)
#         coords2d = coords2d.unsqueeze(0).repeat(B, 1, 1, 1)                                              # (B, 2, H, W)
#         coords3d = torch.cat((coords2d * coords_d, coords_d, torch.ones_like(coords_d)), dim=1)          # (B, 3, H, W)

#         ego2imgs = []
#         for img_meta in img_metas:
#             ego2img = np.eye(4)                                                                          # (4, 4)
#             ego2img[:3] = img_meta['calibration']['trans_ego2img']                                       # (3, 4)
#             ego2imgs.append(ego2img)
#         ego2imgs = np.array(ego2imgs, dtype=np.float32)                                                  # (B, 4, 4)
#         ego2imgs = torch.from_numpy(ego2imgs).to(device)                                                 # (B, 4, 4)
#         coords3d = torch.inverse(ego2imgs).bmm(coords3d.reshape(B, 4, -1))                               # (B, 4, H * W)
#         coords3d = coords3d.reshape(B, 4, H, W)[:, :-1]                                                  # (B, 3, H, W)
#         img_embd = self.position_encoder(coords3d)                                                       # (B, C, H, W)

#         B, C, H, W = bev_feat.size()
#         coords_z = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)                         # (B, 1, H, W)
#         coords_y = torch.linspace(0, bev_h, H, dtype=torch.float32, device=device)                       # (H,)
#         coords_x = torch.linspace(0, bev_w, W, dtype=torch.float32, device=device)                       # (W,)
#         coords2d = torch.stack(torch.meshgrid(coords_x, coords_y, indexing='xy'), dim=0)                 # (2, H, W)
#         coords2d = coords2d.unsqueeze(0).repeat(B, 1, 1, 1)                                              # (B, 2, H, W)
#         coords3d = torch.cat((coords2d, coords_z, torch.ones_like(coords_z)), dim=1)                     # (B, 4, H, W)

#         bev2egos = []
#         for img_meta in img_metas:
#             bev2egos.append(img_meta['calibration']['trans_bev2ego'])
#         bev2egos = np.array(bev2egos, dtype=np.float32)                                                  # (B, 4, 4)
#         bev2egos = torch.from_numpy(bev2egos).to(device)                                                 # (B, 4, 4)
#         coords3d = bev2egos.bmm(coords3d.reshape(B, 4, -1))                                              # (B, 4, H * W)
#         coords3d = coords3d.reshape(B, 4, H, W)[:, :-1]                                                  # (B, 3, H, W)
#         bev_embd = self.position_encoder(coords3d)                                                       # (B, C, H, W)
#         return bev_embd, img_embd


class PositionEmbeddingProj(nn.Module):

    def __init__(self,
                 embed_dim=256,
                 x_bound=[5, 65, 0.234375],
                 y_bound=[-15, 15, 0.234375],
                 z_bound=[-1, 1, 0.2],
                 d_bound=[3.0, 68.0, 0.2],
                 LID=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.d_bound = d_bound
        self.LID = LID

        self.d_num = int((d_bound[1] - d_bound[0]) / d_bound[2])
        self.x_num = int((x_bound[1] - x_bound[0]) / x_bound[2])
        self.y_num = int((y_bound[1] - y_bound[0]) / y_bound[2])
        self.z_num = int((z_bound[1] - z_bound[0]) / z_bound[2])

        self.d_head = nn.Conv2d(self.embed_dim, self.d_num, kernel_size=1)
        self.z_head = nn.Conv2d(self.embed_dim, self.z_num, kernel_size=1)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dim // 4),
            nn.Linear(self.embed_dim // 4, self.embed_dim),
        )

    def get_embeds_2d(self, img_feat, ego2imgs, img_shape):
        B, C, H, W = img_feat.size()
        img_h, img_w = img_shape[:2]
        device = img_feat.device

        coords_h = torch.arange(H, dtype=torch.float32, device=device) * img_h / H
        coords_w = torch.arange(W, dtype=torch.float32, device=device) * img_w / W
        if self.LID:
            index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
            index_1 = index + 1
            bin_size = (self.d_bound[1] - self.d_bound[0]) / (self.d_num * (1 + self.d_num))
            coords_d = self.d_bound[0] + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
            bin_size = (self.d_bound[1] - self.d_bound[0]) / self.d_num
            coords_d = self.d_bound[0] + bin_size * index

        D = coords_d.shape[0]
        coords2d = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij'), dim=-1)   # (W, H, D, 3)
        coords2d = torch.cat((coords2d, torch.ones_like(coords2d[..., :1])), dim=-1)                    # (W, H, D, 4)
        coords2d[..., :2] = coords2d[..., :2] * coords2d[..., 2:3]                                      # (W, H, D, 4)

        img2egos = torch.inverse(ego2imgs)                                                              # (B, 4, 4)
        coords2d = coords2d.view(1, W, H, D, 4, 1).repeat(B, 1, 1, 1, 1, 1)                             # (B, W, H, D, 4, 1)
        img2egos = img2egos.view(B, 1, 1, 1, 4, 4).repeat(1, W, H, D, 1, 1)                             # (B, W, H, D, 4, 4)
        coords2d = torch.matmul(img2egos, coords2d).squeeze(-1)[..., :3]                                # (B, W, H, D, 3)
        embeds2d = self.position_encoder[0](coords2d)                                                   # (B, W, H, D, 64)

        depths2d = self.d_head(img_feat).softmax(dim=1)                                                 # (B, D, H, W)
        depths2d = depths2d.permute(0, 3, 2, 1).unsqueeze(-1)                                           # (B, W, H, D, 1)
        embeds2d = (embeds2d * depths2d).sum(dim=3)                                                     # (B, W, H, 64)
        embeds2d = self.position_encoder[1](embeds2d)                                                   # (B, W, H, 256)
        embeds2d = embeds2d.permute(0, 3, 2, 1).contiguous()                                            # (B, 256, H, W)
        return embeds2d

    def get_embeds_3d(self, bev_feat):
        device = bev_feat.device
        B, C, H, W = bev_feat.size()

        coords_x = torch.arange(*self.x_bound, dtype=torch.float32, device=device)
        coords_y = torch.arange(*self.y_bound, dtype=torch.float32, device=device)
        coords_z = torch.arange(*self.z_bound, dtype=torch.float32, device=device)
        coords3d = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z], indexing='ij'), dim=-1)   # (X, Y, Z, 3)

        X, Y, Z = coords3d.shape[:3]
        coords3d = coords3d.view(1, X, Y, Z, 3).repeat(B, 1, 1, 1, 1)                                   # (B, X, Y, Z, 3)
        embeds3d = self.position_encoder[0](coords3d)                                                   # (B, X, Y, Z, 64)

        height3d = self.z_head(bev_feat).softmax(dim=1)                                                 # (B, Z, Y, X)
        height3d = height3d.permute(0, 3, 2, 1).unsqueeze(-1)                                           # (B, X, Y, Z, 1)
        embeds3d = (embeds3d * height3d).sum(dim=3)                                                     # (B, X, Y, 64)
        embeds3d = self.position_encoder[1](embeds3d)                                                   # (B, X, Y, 256)
        embeds3d = embeds3d.permute(0, 3, 2, 1).contiguous()                                            # (B, 256, Y, X)
        return embeds3d

    def forward(self, bev_feat, img_feat, depth_prior, img_metas):
        img_shape = img_metas[0]['input_shape']

        ego2imgs = []
        for img_meta in img_metas:
            ego2img = np.eye(4)                                                                         # (4, 4)
            ego2img[:3] = img_meta['calibration']['trans_ego2img']                                      # (3, 4)
            ego2imgs.append(ego2img)
        ego2imgs = np.array(ego2imgs, dtype=np.float32)                                                 # (B, 4, 4)
        ego2imgs = torch.from_numpy(ego2imgs).to(img_feat.device)                                       # (B, 4, 4)

        embeds2d = self.get_embeds_2d(img_feat, ego2imgs, img_shape)
        embeds3d = self.get_embeds_3d(bev_feat)
        return embeds3d, embeds2d
