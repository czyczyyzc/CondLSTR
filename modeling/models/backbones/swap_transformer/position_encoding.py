"""
Various positional encodings for the transformer.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats // 2, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.num_pos_feats // 2))

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos=(50, 50), num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(num_pos[0], num_pos_feats // 2)
        self.col_embed = nn.Embedding(num_pos[1], num_pos_feats // 2)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.uniform_(self.row_embed.weight)
        # nn.init.uniform_(self.col_embed.weight)
        nn.init.normal_(self.row_embed.weight)
        nn.init.normal_(self.col_embed.weight)

    def forward(self, mask):
        h, w = mask.shape[-2:]
        i = torch.arange(w, device=mask.device)
        j = torch.arange(h, device=mask.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


class PositionEmbeddingAnchor(nn.Module):

    def __init__(self, num_query_position=300, num_query_pattern=3, d_model=256):
        super().__init__()
        self.num_position = num_query_position
        self.num_pattern = num_query_pattern
        self.d_model = d_model

        self.position = nn.Embedding(self.num_position, 2)
        self.pattern = nn.Embedding(self.num_pattern, self.d_model)

        self.adapt_pos2d = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.position.weight.data, 0, 1)

    @staticmethod
    def mask2pos(mask):
        not_mask = ~mask
        y_embed = not_mask[:, :, 0].cumsum(1, dtype=torch.float32)
        x_embed = not_mask[:, 0, :].cumsum(1, dtype=torch.float32)
        y_embed = (y_embed - 0.5) / y_embed[:, -1:]
        x_embed = (x_embed - 0.5) / x_embed[:, -1:]
        return y_embed, x_embed

    @staticmethod
    def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        return posemb

    def forward(self, mask):
        bs, h, w = mask.shape
        reference_points = self.position.weight.unsqueeze(0).repeat(bs, self.num_pattern, 1)
        tgt = self.pattern.weight.reshape(1, self.num_pattern, 1, self.d_model).repeat(
            bs, 1, self.num_position, 1).reshape(bs, self.num_pattern * self.num_position, self.d_model)
        pos_col, pos_row = self.mask2pos(mask)
        pos_2d = torch.cat([pos_row.unsqueeze(1).repeat(1, h, 1).unsqueeze(-1),
                            pos_col.unsqueeze(2).repeat(1, 1, w).unsqueeze(-1)], dim=-1)
        posemb_2d = self.adapt_pos2d(self.pos2posemb2d(pos_2d))
        query_pos = self.adapt_pos2d(self.pos2posemb2d(reference_points))
        posemb_2d = posemb_2d.permute(0, 3, 1, 2).contiguous()
        query_pos = query_pos.permute(0, 2, 1).contiguous().unsqueeze(-1)
        tgt = tgt.permute(0, 2, 1).contiguous().unsqueeze(-1)
        return posemb_2d, query_pos, tgt


class PositionEmbeddingLearnedV2(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos=(50, 50), num_pos_feats=256):
        super().__init__()
        self.num_pos = num_pos
        self.pos_embed = nn.Embedding(num_pos[0] * num_pos[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.pos_embed.weight)

    def forward(self, mask):
        h, w = mask.shape[-2:]
        pos = self.pos_embed.weight.view(*self.num_pos, -1)[:h, :w]
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# class PositionEmbeddingProj(nn.Module):
#
#     def __init__(self,
#                  embed_dim=256,
#                  x_bound=[5, 65, 0.234375],
#                  y_bound=[-15, 15, 0.234375],
#                  z_bound=[-1, 1, 0.2],
#                  d_bound=[3.0, 68.0, 0.2],
#                  normalize=True,
#                  LID=False):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.x_bound = x_bound
#         self.y_bound = y_bound
#         self.z_bound = z_bound
#         self.d_bound = d_bound
#         self.normalize = normalize
#         self.LID = LID
#
#         self.d_num = int((d_bound[1] - d_bound[0]) / d_bound[2])
#         self.x_num = int((x_bound[1] - x_bound[0]) / x_bound[2])
#         self.y_num = int((y_bound[1] - y_bound[0]) / y_bound[2])
#         self.z_num = int((z_bound[1] - z_bound[0]) / z_bound[2])
#
#         self.position_encoder = nn.Sequential(
#             nn.Conv2d(self.d_num * 3, self.embed_dim * 4, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(self.embed_dim * 4, self.embed_dim, kernel_size=1, stride=1, padding=0),
#         )
#
#     def forward(self, img_feat, cam2imgs=None, pad_shape=None):
#         pad_h, pad_w = pad_shape[:2]
#         if len(img_feat.shape) == 4:
#             B, C, H, W = img_feat.shape
#             N = 1
#         else:
#             B, N, C, H, W = img_feat.shape
#         device = img_feat.device
#
#         coords_h = torch.arange(H, dtype=torch.float32, device=device) * pad_h / H
#         coords_w = torch.arange(W, dtype=torch.float32, device=device) * pad_w / W
#
#         if self.LID:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             index_1 = index + 1
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / (self.d_num * (1 + self.d_num))
#             coords_d = self.d_bound[0] + bin_size * index * index_1
#         else:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / self.d_num
#             coords_d = self.d_bound[0] + bin_size * index
#
#         D = coords_d.shape[0]
#         coords2d = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d]), dim=-1)        # (W, H, D, 3)
#         coords2d[..., :2] = coords2d[..., :2] * coords2d[..., 2:3]                            # (W, H, D, 3)
#
#         img2cams = torch.inverse(cam2imgs)                                                    # (B, N, 3, 3)
#         coord_2d = coords2d.view(1, 1, W, H, D, 3, 1).repeat(B, N, 1, 1, 1, 1, 1)             # (B, N, W, H, D, 3, 1)
#         img2cams = img2cams.view(B, N, 1, 1, 1, 3, 3).repeat(1, 1, W, H, D, 1, 1)             # (B, N, W, H, D, 3, 3)
#         coord_2d = torch.matmul(img2cams, coord_2d).squeeze(-1)[..., :3]                      # (B, N, W, H, D, 3)
#         coord_2d = coord_2d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, D * 3, H, W)   # (B * N, D * 3, H, W)
#         embeds2d = self.position_encoder(coord_2d)                                            # (B * N, C, H, W)
#         return embeds2d


class PositionEmbeddingProj(nn.Module):

    def __init__(self,
                 in_channels,
                 embed_dim=256,
                 x_bound=[5, 65, 0.234375],
                 y_bound=[-15, 15, 0.234375],
                 z_bound=[-1, 1, 0.2],
                 d_bound=[3.0, 68.0, 0.2],
                 LID=False,
                 use_input_proj=True,
                 with_3d=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.d_bound = d_bound
        self.LID = LID
        self.use_input_proj = use_input_proj
        self.with_3d = with_3d

        self.d_num = int((d_bound[1] - d_bound[0]) / d_bound[2])
        self.x_num = int((x_bound[1] - x_bound[0]) / x_bound[2])
        self.y_num = int((y_bound[1] - y_bound[0]) / y_bound[2])
        self.z_num = int((z_bound[1] - z_bound[0]) / z_bound[2])

        if self.use_input_proj:
            self.input_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)
        else:
            self.input_proj = nn.Identity()

        self.d_head = nn.Conv2d(self.embed_dim, self.d_num, kernel_size=1)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dim // 4),
            nn.Linear(self.embed_dim // 4, self.embed_dim),
        )

        if self.with_3d:
            self.z_head = nn.Conv2d(self.embed_dim, self.z_num, kernel_size=1)
            self.map_feat = nn.Parameter(torch.randn((1, self.embed_dim, self.y_num, self.x_num)))
            nn.init.normal_(self.map_feat)

    def get_embeds_2d(self, img_feat, ego2imgs, pad_shape):
        B, C, H, W = img_feat.size()
        pad_h, pad_w = pad_shape[:2]
        device = img_feat.device

        coords_h = torch.arange(H, dtype=torch.float32, device=device) * pad_h / H
        coords_w = torch.arange(W, dtype=torch.float32, device=device) * pad_w / W
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
        coords2d = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d]), dim=-1)      # (W, H, D, 3)
        coords2d = torch.cat((coords2d, torch.ones_like(coords2d[..., :1])), dim=-1)        # (W, H, D, 4)
        coords2d[..., :2] = coords2d[..., :2] * coords2d[..., 2:3]                          # (W, H, D, 4)

        ego2imgs = F.pad(ego2imgs, (0, 0, 0, 1), mode='constant', value=0.0)                # (B, 4, 4)
        ego2imgs[..., -1, -1] = 1.0                                                         # (B, 4, 4)
        img2egos = torch.inverse(ego2imgs)                                                  # (B, 4, 4)

        coords2d = coords2d.view(1, W, H, D, 4, 1).repeat(B, 1, 1, 1, 1, 1)                 # (B, W, H, D, 4, 1)
        img2egos = img2egos.view(B, 1, 1, 1, 4, 4).repeat(1, W, H, D, 1, 1)                 # (B, W, H, D, 4, 4)
        coords2d = torch.matmul(img2egos, coords2d).squeeze(-1)[..., :3]                    # (B, W, H, D, 3)
        embeds2d = self.position_encoder[0](coords2d)                                       # (B, W, H, D, 64)

        depths2d = self.d_head(img_feat).softmax(dim=1)                                     # (B, D, H, W)
        depths2d = depths2d.permute(0, 3, 2, 1).unsqueeze(-1)                               # (B, W, H, D, 1)
        embeds2d = (embeds2d * depths2d).sum(dim=3)                                         # (B, W, H, 64)
        embeds2d = self.position_encoder[1](embeds2d)                                       # (B, W, H, 256)
        embeds2d = embeds2d.permute(0, 3, 2, 1).contiguous()                                # (B, 256, H, W)
        return embeds2d

    def get_embeds_3d(self, map_feat):
        device = map_feat.device
        B, C, H, W = map_feat.size()

        coords_x = torch.arange(*self.x_bound, dtype=torch.float32, device=device)
        coords_y = torch.arange(*self.y_bound, dtype=torch.float32, device=device)
        coords_z = torch.arange(*self.z_bound, dtype=torch.float32, device=device)
        coords3d = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z]), dim=-1)      # (X, Y, Z, 3)
        coords3d = coords3d.flip([1])                                                       # (X, Y, Z, 3)

        X, Y, Z = coords3d.shape[:3]
        coords3d = coords3d.view(1, X, Y, Z, 3).repeat(B, 1, 1, 1, 1)                       # (B, X, Y, Z, 3)
        embeds3d = self.position_encoder[0](coords3d)                                       # (B, X, Y, Z, 64)

        height3d = self.z_head(map_feat).softmax(dim=1)                                     # (B, Z, Y, X)
        height3d = height3d.permute(0, 3, 2, 1).unsqueeze(-1)                               # (B, X, Y, Z, 1)
        embeds3d = (embeds3d * height3d).sum(dim=3)                                         # (B, X, Y, 64)
        embeds3d = self.position_encoder[1](embeds3d)                                       # (B, X, Y, 256)
        embeds3d = embeds3d.permute(0, 3, 2, 1).contiguous()                                # (B, 256, Y, X)
        return embeds3d

    def forward(self, img_feat, ego2imgs=None, pad_shape=None):
        img_feat = self.input_proj(img_feat)
        embeds2d = self.get_embeds_2d(img_feat, ego2imgs, pad_shape)
        if self.with_3d:
            map_feat = self.map_feat.repeat(img_feat.shape[0], 1, 1, 1)
            embeds3d = self.get_embeds_3d(map_feat)
            return embeds2d, embeds3d, img_feat, map_feat
        else:
            return embeds2d, img_feat


class PositionEmbeddingProjV2(nn.Module):

    def __init__(self,
                 in_channels,
                 embed_dim=256,
                 x_bound=[5, 65, 0.234375],
                 y_bound=[-15, 15, 0.234375],
                 z_bound=[-1, 1, 0.2],
                 d_bound=[3.0, 68.0, 0.2],
                 normalize=True,
                 LID=False,
                 with_z=True,
                 src_shape=None,
                 pad_shape=None,
                 ego2imgs=None,
                 use_fix_encode=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.d_bound = d_bound
        self.normalize = normalize
        self.LID = LID
        self.with_z = with_z
        self.use_fix_encode = use_fix_encode

        self.d_num = int((d_bound[1] - d_bound[0]) / d_bound[2])
        self.x_num = int((x_bound[1] - x_bound[0]) / x_bound[2])
        self.y_num = int((y_bound[1] - y_bound[0]) / y_bound[2])
        self.z_num = int((z_bound[1] - z_bound[0]) / z_bound[2])

        self.input_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)

        self.d_head = nn.Conv2d(self.embed_dim, self.d_num, kernel_size=1)

        if self.with_z:
            self.z_head = nn.Conv2d(self.embed_dim, self.z_num, kernel_size=1)
            self.map_feat = nn.Parameter(torch.randn((1, self.embed_dim, self.x_num, self.y_num)))
            nn.init.normal_(self.map_feat)

        self.position_encoder = nn.Sequential(
            nn.Conv2d(3, self.embed_dim * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim * 4, self.embed_dim, kernel_size=1, stride=1, padding=0),
        )

        if self.use_fix_encode:
            B, N = ego2imgs.shape[:2]
            H, W = src_shape[:2]
            pad_h, pad_w = pad_shape[:2]
            coords2d = self.get_coords_2d(1, N, H, W, pad_h, pad_w, ego2imgs)
            self.register_buffer('coords2d', coords2d)
            if self.with_z:
                coords3d = self.get_coords_3d(1)
                self.register_buffer('coords3d', coords3d)

    def get_coords_2d(self, B, N, H, W, pad_h, pad_w, ego2imgs, device='cpu'):
        coords_h = torch.arange(H, dtype=torch.float32, device=device) * pad_h / H
        coords_w = torch.arange(W, dtype=torch.float32, device=device) * pad_w / W

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
        coords2d = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d]), dim=-1)      # (W, H, D, 3)
        coords2d = torch.cat((coords2d, torch.ones_like(coords2d[..., :1])), dim=-1)        # (W, H, D, 3)
        coords2d[..., :2] = coords2d[..., :2] * coords2d[..., 2:3]

        if ego2imgs.shape[-1] == 3:
            ego2imgs = F.pad(ego2imgs, (0, 1, 0, 1), mode='constant', value=0.0)            # (B, N, 4, 4)
            ego2imgs[..., -1, -1] = 1.0                                                     # (B, N, 4, 4)

        img2egos = torch.inverse(ego2imgs)                                                  # (B, N, 4, 4)
        coords2d = coords2d.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)           # (B, N, W, H, D, 4, 1)
        img2egos = img2egos.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)           # (B, N, W, H, D, 4, 4)
        coords2d = torch.matmul(img2egos, coords2d).squeeze(-1)[..., :3]                    # (B, N, W, H, D, 3)
        coords2d = coords2d.view(B * N, W, H, D, 3).permute(0, 3, 2, 1, 4)                  # (B * N, D, H, W, 3)
        return coords2d

    def get_coords_3d(self, B, device='cpu'):
        coords_x = torch.arange(*self.x_bound, dtype=torch.float32, device=device)
        coords_y = torch.arange(*self.y_bound, dtype=torch.float32, device=device)
        coords_z = torch.arange(*self.z_bound, dtype=torch.float32, device=device)
        coords3d = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z]), dim=-1)      # (X, Y, Z, 3)

        X, Y, Z = coords3d.shape[:3]
        coords3d = coords3d.view(1, X, Y, Z, 3).repeat(B, 1, 1, 1, 1)                       # (B, X, Y, Z, 3)
        coords3d = coords3d.permute(0, 3, 1, 2, 4)                                          # (B, Z, X, Y, 3)
        return coords3d

    def forward(self, img_feat, ego2imgs=None, pad_shape=None):
        pad_h, pad_w = pad_shape[:2]
        if len(img_feat.shape) == 4:
            B, C, H, W = img_feat.shape
            N = 1
        else:
            B, N, C, H, W = img_feat.shape
        device = img_feat.device

        if self.use_fix_encode:
            coords2d = self.coords2d.repeat(B, 1, 1, 1, 1)                                  # (B * N, D, H, W, 3)
        else:
            coords2d = self.get_coords_2d(B, N, H, W, pad_h, pad_w, ego2imgs, device)       # (B * N, D, H, W, 3)

        img_feat = img_feat.view(B * N, -1, H, W)                                           # (B * N, C, H, W)
        img_feat = self.input_proj(img_feat)                                                # (B * N, C, H, W)
        img_prob = self.d_head(img_feat).softmax(dim=1).unsqueeze(4)                        # (B * N, D, H, W, 1)
        coords2d = (coords2d * img_prob).sum(dim=1)                                         # (B * N, H, W, 3)
        coords2d = coords2d.permute(0, 3, 1, 2).contiguous()                                # (B * N, 3, H, W)

        embeds2d = self.position_encoder(coords2d)                                          # (B * N, C, H, W)
        if N > 1:
            embeds2d = embeds2d.view(B, N, self.embed_dim, H, W)

        if self.with_z:
            if self.use_fix_encode:
                coords3d = self.coords3d.repeat(B, 1, 1, 1, 1)                              # (B, Z, X, Y, 3)
            else:
                coords3d = self.get_coords_3d(B, device)                                    # (B, Z, X, Y, 3)

            map_feat = self.map_feat.repeat(B, 1, 1, 1)                                     # (B, C, X, Y)
            map_prob = self.z_head(map_feat).softmax(dim=1).unsqueeze(4)                    # (B, Z, X, Y, 1)
            coords3d = (coords3d * map_prob).sum(dim=1)                                     # (B, X, Y, 3)
            coords3d = coords3d.permute(0, 3, 1, 2).flip([2, 3]).contiguous()               # (B, 3, X, Y)

            embeds3d = self.position_encoder(coords3d)
            return embeds2d, embeds3d, img_feat, map_feat
        else:
            return embeds2d, img_feat


class PositionEmbeddingProjV3(nn.Module):

    def __init__(self,
                 in_channels,
                 embed_dim=256,
                 x_bound=[5, 65, 0.9375],
                 y_bound=[-15, 15, 0.9375],
                 z_bound=[-3, 3, 1],
                 d_bound=[0, 90, 0.9375],
                 normalize=True,
                 LID=False,
                 src_shape=None,
                 pad_shape=None,
                 ego2imgs=None,
                 use_fix_encode=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.d_bound = d_bound
        self.normalize = normalize
        self.LID = LID
        self.use_fix_encode = use_fix_encode

        self.d_num = int((d_bound[1] - d_bound[0]) / d_bound[2])
        self.x_num = int((x_bound[1] - x_bound[0]) / x_bound[2])
        self.y_num = int((y_bound[1] - y_bound[0]) / y_bound[2])
        self.z_num = int((z_bound[1] - z_bound[0]) / z_bound[2])

        self.depth_net = nn.Conv2d(in_channels, self.d_num, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.randn((1, self.embed_dim, self.x_num, self.y_num, self.z_num)))
        nn.init.normal_(self.pos_embed)

        if self.use_fix_encode:
            B, N = ego2imgs.shape[:2]
            H, W = src_shape[:2]
            pad_h, pad_w = pad_shape[:2]
            coords2d = self.get_coords_2d(B, N, H, W, pad_h, pad_w)
            self.register_buffer('coords2d', coords2d)
            self.register_buffer('ego2imgs', ego2imgs)

    def get_coords_2d(self, B, N, H, W, pad_h, pad_w, device='cpu'):
        coords_h = torch.arange(H, dtype=torch.float32, device=device) * pad_h / H
        coords_w = torch.arange(W, dtype=torch.float32, device=device) * pad_w / W

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
        coords2d = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d]), dim=-1)       # (W, H, D, 3)
        coords2d = torch.cat((coords2d, torch.ones_like(coords2d[..., :1])), dim=-1)         # (W, H, D, 4)
        coords2d[..., :2] = coords2d[..., :2] * coords2d[..., 2:3]

        coords2d = coords2d.view(1, 1, W, H, D, 4).repeat(B, N, 1, 1, 1, 1)                  # (B, N, W, H, D, 4)
        coords2d = coords2d.view(B * N, W, H, D, 4).permute(0, 3, 2, 1, 4).contiguous()      # (B * N, D, H, W, 4)
        return coords2d

    def position_encoder(self, coords2d):
        """
        pos_embd: (1, C, X, Y, Z)
        coords2d: (B * N, 1, H, W, 3)
        """
        coords2d[..., 0] = ((coords2d[..., 0] - self.x_bound[0]) / (self.x_bound[1] - self.x_bound[0])) * 2 - 1
        coords2d[..., 1] = ((coords2d[..., 1] - self.y_bound[0]) / (self.y_bound[1] - self.y_bound[0])) * 2 - 1
        coords2d[..., 2] = ((coords2d[..., 2] - self.z_bound[0]) / (self.z_bound[1] - self.z_bound[0])) * 2 - 1

        pos_embd = self.pos_embed.expand(coords2d.size(0), -1, -1, -1, -1)                   # (B * N, C, X, Y, Z)
        embeds2d = pos_embd.flip([2, 3])                                                     # (B * N, C, X, Y, Z)
        embeds2d = F.grid_sample(embeds2d, coords2d, mode='bilinear', padding_mode='zeros')  # (B * N, C, 1, H, W)
        embeds2d = embeds2d.squeeze(2)                                                       # (B * N, C, H, W)
        return embeds2d

    def forward(self, img_feat, ego2imgs=None, pad_shape=None, src_mask=None):
        pad_h, pad_w = pad_shape[:2]
        if len(img_feat.shape) == 4:
            B, C, H, W = img_feat.shape
            N = 1
        else:
            B, N, C, H, W = img_feat.shape
        device = img_feat.device

        if self.use_fix_encode:
            coords2d = self.coords2d                                                         # (B * N, D, H, W, 4)
            ego2imgs = self.ego2imgs                                                         # (B, N, 4, 4)
            src_embd = self.src_embed.repeat(B * N, 1, 1, 1)                                 # (B * N, C, H, W)
        else:
            coords2d = self.get_coords_2d(B, N, H, W, pad_h, pad_w, device)                  # (B * N, D, H, W, 4)
            if src_mask is None:
                src_mask = torch.zeros((B * N, H, W), dtype=torch.bool, device=device)       # (B * N, H, W)
            src_embd = self.src_embed(src_mask)                                              # (B * N, C, H, W)

        img_feat = img_feat.view(B * N, -1, H, W)                                            # (B * N, C, H, W)
        img_prob = self.depth_net(img_feat).softmax(dim=1).unsqueeze(4)                      # (B * N, D, H, W, 1)
        coords2d = (coords2d * img_prob).sum(dim=1).unsqueeze(4)                             # (B * N, H, W, 4, 1)

        if ego2imgs.shape[-1] == 3:
            ego2imgs = F.pad(ego2imgs, (0, 1, 0, 1), mode='constant', value=0.0)             # (B, N, 4, 4)
            ego2imgs[..., -1, -1] = 1.0                                                      # (B, N, 4, 4)

        img2egos = torch.inverse(ego2imgs)                                                   # (B, N, 4, 4)
        img2egos = img2egos.view(B * N, 1, 1, 4, 4).repeat(1, H, W, 1, 1)                    # (B * N, H, W, 4, 4)
        coords2d = torch.matmul(img2egos, coords2d).squeeze(-1)[..., :3]                     # (B * N, H, W, 3)
        coords2d = coords2d.unsqueeze(1)                                                     # (B * N, 1, H, W, 3)

        embeds3d = self.pos_embed.mean(dim=4).repeat(B, 1, 1, 1)                             # (B, C, X, Y)
        embeds2d = self.position_encoder(coords2d)                                           # (B * N, C, H, W)
        embeds2d = embeds2d + src_embd
        if N > 1:
            embeds2d = embeds2d.view(B, N, self.embed_dim, H, W)
        return embeds2d, embeds3d


# class PositionEmbeddingProjV2(nn.Module):
#
#     def __init__(self,
#                  in_channels,
#                  embed_dim=256,
#                  x_bound=[5, 65, 0.234375],
#                  y_bound=[-15, 15, 0.234375],
#                  z_bound=[-1, 1, 0.2],
#                  d_bound=[3.0, 68.0, 0.2],
#                  normalize=True,
#                  LID=False,
#                  with_z=True,
#                  src_shape=None,
#                  pad_shape=None,
#                  use_fix_encode=False):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.x_bound = x_bound
#         self.y_bound = y_bound
#         self.z_bound = z_bound
#         self.d_bound = d_bound
#         self.normalize = normalize
#         self.LID = LID
#         self.with_z = with_z
#         self.use_fix_encode = use_fix_encode
#
#         self.d_num = int((d_bound[1] - d_bound[0]) / d_bound[2])
#         self.x_num = int((x_bound[1] - x_bound[0]) / x_bound[2])
#         self.y_num = int((y_bound[1] - y_bound[0]) / y_bound[2])
#         self.z_num = int((z_bound[1] - z_bound[0]) / z_bound[2])
#
#         self.input_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)
#
#         self.d_head = nn.Conv2d(self.embed_dim, self.d_num, kernel_size=1)
#
#         if self.with_z:
#             self.z_head = nn.Conv2d(self.embed_dim, self.z_num, kernel_size=1)
#             self.map_feat = nn.Parameter(torch.randn((1, self.embed_dim, self.x_num, self.y_num)))
#             nn.init.normal_(self.map_feat)
#
#         self.position_encoder = nn.Sequential(
#             nn.Conv2d(3, self.embed_dim * 4, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(self.embed_dim * 4, self.embed_dim, kernel_size=1, stride=1, padding=0),
#         )
#
#         if self.use_fix_encode:
#             H, W = src_shape[:2]
#             pad_h, pad_w = pad_shape[:2]
#             coords2d = self.get_coords_2d(1, 1, H, W, pad_h, pad_w)
#             self.register_buffer('coords2d', coords2d)
#             if self.with_z:
#                 coords3d = self.get_coords_3d(1)
#                 self.register_buffer('coords3d', coords3d)
#
#     def get_coords_2d(self, B, N, H, W, pad_h, pad_w, device='cpu'):
#         coords_h = torch.arange(H, dtype=torch.float32, device=device) * pad_h / H
#         coords_w = torch.arange(W, dtype=torch.float32, device=device) * pad_w / W
#
#         if self.LID:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             index_1 = index + 1
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / (self.d_num * (1 + self.d_num))
#             coords_d = self.d_bound[0] + bin_size * index * index_1
#         else:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / self.d_num
#             coords_d = self.d_bound[0] + bin_size * index
#
#         D = coords_d.shape[0]
#         coords2d = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d]), dim=-1)       # (W, H, D, 3)
#         coords2d = torch.cat((coords2d, torch.ones_like(coords2d[..., :1])), dim=-1)         # (W, H, D, 4)
#         coords2d[..., :2] = coords2d[..., :2] * coords2d[..., 2:3]
#
#         coords2d = coords2d.view(1, 1, W, H, D, 4).repeat(B, N, 1, 1, 1, 1)                  # (B, N, W, H, D, 4)
#         coords2d = coords2d.view(B * N, W, H, D, 4).permute(0, 3, 2, 1, 4).contiguous()      # (B * N, D, H, W, 4)
#         return coords2d
#
#     def get_coords_3d(self, B, device='cpu'):
#         coords_x = torch.arange(*self.x_bound, dtype=torch.float32, device=device)
#         coords_y = torch.arange(*self.y_bound, dtype=torch.float32, device=device)
#         coords_z = torch.arange(*self.z_bound, dtype=torch.float32, device=device)
#         coords3d = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z]), dim=-1)       # (X, Y, Z, 3)
#
#         X, Y, Z = coords3d.shape[:3]
#         coords3d = coords3d.view(1, X, Y, Z, 3).repeat(B, 1, 1, 1, 1)                        # (B, X, Y, Z, 3)
#         coords3d = coords3d.permute(0, 3, 1, 2, 4)                                           # (B, Z, X, Y, 3)
#         return coords3d
#
#     def forward(self, img_feat, ego2imgs=None, pad_shape=None):
#         pad_h, pad_w = pad_shape[:2]
#         if len(img_feat.shape) == 4:
#             B, C, H, W = img_feat.shape
#             N = 1
#         else:
#             B, N, C, H, W = img_feat.shape
#         device = img_feat.device
#
#         if self.use_fix_encode:
#             coords2d = self.coords2d.repeat(B, 1, 1, 1, 1)                                   # (B * N, D, H, W, 4)
#         else:
#             coords2d = self.get_coords_2d(B, N, H, W, pad_h, pad_w, device)                  # (B * N, D, H, W, 4)
#
#         img_feat = img_feat.view(B * N, -1, H, W)                                            # (B * N, C, H, W)
#         img_feat = self.input_proj(img_feat)                                                 # (B * N, C, H, W)
#
#         img_prob = self.d_head(img_feat).softmax(dim=1).unsqueeze(4)                         # (B * N, D, H, W, 1)
#         coords2d = (coords2d * img_prob).sum(dim=1).unsqueeze(4)                             # (B * N, H, W, 4, 1)
#
#         if ego2imgs.shape[-1] == 3:
#             ego2imgs = F.pad(ego2imgs, (0, 1, 0, 1), mode='constant', value=0.0)             # (B, N, 4, 4)
#             ego2imgs[..., -1, -1] = 1.0                                                      # (B, N, 4, 4)
#
#         img2egos = torch.inverse(ego2imgs)                                                   # (B, N, 4, 4)
#         img2egos = img2egos.view(B * N, 1, 1, 4, 4).repeat(1, H, W, 1, 1)                    # (B * N, H, W, 4, 4)
#         coords2d = torch.matmul(img2egos, coords2d).squeeze(-1)[..., :3]                     # (B * N, H, W, 3)
#         coords2d = coords2d.permute(0, 3, 1, 2).contiguous()                                 # (B * N, 3, H, W)
#
#         embeds2d = self.position_encoder(coords2d)                                           # (B * N, C, H, W)
#         embeds2d = embeds2d.view(B, N, self.embed_dim, H, W)
#         if len(img_feat.shape) == 4:
#             embeds2d = embeds2d.squeeze(1)
#
#         if self.with_z:
#             if self.use_fix_encode:
#                 coords3d = self.coords3d.repeat(B, 1, 1, 1, 1)                              # (B, Z, X, Y, 3)
#             else:
#                 coords3d = self.get_coords_3d(B, device)                                    # (B, Z, X, Y, 3)
#
#             map_feat = self.map_feat.repeat(B, 1, 1, 1)                                     # (B, C, X, Y)
#             map_prob = self.z_head(map_feat).softmax(dim=1).unsqueeze(4)                    # (B, Z, X, Y, 1)
#             coords3d = (coords3d * map_prob).sum(dim=1)                                     # (B, X, Y, 3)
#             coords3d = coords3d.permute(0, 3, 1, 2).flip([2, 3]).contiguous()               # (B, 3, X, Y)
#
#             embeds3d = self.position_encoder(coords3d)
#             return embeds2d, embeds3d, img_feat, map_feat
#         else:
#             return embeds2d, img_feat


# class PositionEmbeddingProjV3(nn.Module):
#
#     def __init__(self,
#                  in_channels,
#                  embed_dim=256,
#                  x_bound=[5, 65, 0.9375],
#                  y_bound=[-15, 15, 0.9375],
#                  z_bound=[-3, 3, 1],
#                  d_bound=[0, 90, 0.9375],
#                  normalize=True,
#                  LID=False,
#                  src_shape=None,
#                  pad_shape=None,
#                  ego2imgs=None,
#                  use_fix_encode=False):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.x_bound = x_bound
#         self.y_bound = y_bound
#         self.z_bound = z_bound
#         self.d_bound = d_bound
#         self.normalize = normalize
#         self.LID = LID
#         self.use_fix_encode = use_fix_encode
#
#         self.d_num = int((d_bound[1] - d_bound[0]) / d_bound[2])
#         self.x_num = int((x_bound[1] - x_bound[0]) / x_bound[2])
#         self.y_num = int((y_bound[1] - y_bound[0]) / y_bound[2])
#         self.z_num = int((z_bound[1] - z_bound[0]) / z_bound[2])
#
#         self.depth_net = nn.Conv2d(in_channels, self.d_num, kernel_size=1)
#         self.pos_embed = nn.Parameter(torch.randn((1, self.embed_dim, self.x_num, self.y_num, self.z_num)))
#         nn.init.normal_(self.pos_embed)
#
#         self.src_embed = PositionEmbeddingSine(self.embed_dim, normalize=True)
#         if self.use_fix_encode:
#             B, N = ego2imgs.shape[:2]
#             H, W = src_shape[:2]
#             pad_h, pad_w = pad_shape[:2]
#             coords2d = self.get_coords_2d(B, N, H, W, pad_h, pad_w)
#             src_mask = torch.zeros((1, H, W), dtype=torch.bool)
#             self.register_buffer('coords2d', coords2d)
#             self.register_buffer('ego2imgs', ego2imgs)
#             self.src_embed = nn.Parameter(self.src_pos_embed(src_mask), requires_grad=False)
#
#     def get_coords_2d(self, B, N, H, W, pad_h, pad_w, device='cpu'):
#         coords_h = torch.arange(H, dtype=torch.float32, device=device) * pad_h / H
#         coords_w = torch.arange(W, dtype=torch.float32, device=device) * pad_w / W
#
#         if self.LID:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             index_1 = index + 1
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / (self.d_num * (1 + self.d_num))
#             coords_d = self.d_bound[0] + bin_size * index * index_1
#         else:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / self.d_num
#             coords_d = self.d_bound[0] + bin_size * index
#
#         D = coords_d.shape[0]
#         coords2d = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d]), dim=-1)       # (W, H, D, 3)
#         coords2d = torch.cat((coords2d, torch.ones_like(coords2d[..., :1])), dim=-1)         # (W, H, D, 4)
#         coords2d[..., :2] = coords2d[..., :2] * coords2d[..., 2:3]
#
#         coords2d = coords2d.view(1, 1, W, H, D, 4).repeat(B, N, 1, 1, 1, 1)                  # (B, N, W, H, D, 4)
#         coords2d = coords2d.view(B * N, W, H, D, 4).permute(0, 3, 2, 1, 4).contiguous()      # (B * N, D, H, W, 4)
#         return coords2d
#
#     def position_encoder(self, coords2d):
#         """
#         pos_embd: (1, C, X, Y, Z)
#         coords2d: (B * N, 1, H, W, 3)
#         """
#         coords2d[..., 0] = ((coords2d[..., 0] - self.x_bound[0]) / (self.x_bound[1] - self.x_bound[0])) * 2 - 1
#         coords2d[..., 1] = ((coords2d[..., 1] - self.y_bound[0]) / (self.y_bound[1] - self.y_bound[0])) * 2 - 1
#         coords2d[..., 2] = ((coords2d[..., 2] - self.z_bound[0]) / (self.z_bound[1] - self.z_bound[0])) * 2 - 1
#
#         pos_embd = self.pos_embed.expand(coords2d.size(0), -1, -1, -1, -1)                   # (B * N, C, X, Y, Z)
#         embeds2d = pos_embd.flip([2, 3])                                                     # (B * N, C, X, Y, Z)
#         embeds2d = F.grid_sample(embeds2d, coords2d, mode='bilinear', padding_mode='zeros')  # (B * N, C, 1, H, W)
#         embeds2d = embeds2d.squeeze(2)                                                       # (B * N, C, H, W)
#         return embeds2d
#
#     def forward(self, img_feat, ego2imgs=None, pad_shape=None, src_mask=None):
#         pad_h, pad_w = pad_shape[:2]
#         if len(img_feat.shape) == 4:
#             B, C, H, W = img_feat.shape
#             N = 1
#         else:
#             B, N, C, H, W = img_feat.shape
#         device = img_feat.device
#
#         if self.use_fix_encode:
#             coords2d = self.coords2d                                                         # (B * N, D, H, W, 4)
#             ego2imgs = self.ego2imgs                                                         # (B, N, 4, 4)
#             src_embd = self.src_embed.repeat(B * N, 1, 1, 1)                                 # (B * N, C, H, W)
#         else:
#             coords2d = self.get_coords_2d(B, N, H, W, pad_h, pad_w, device)                  # (B * N, D, H, W, 4)
#             if src_mask is None:
#                 src_mask = torch.zeros((B * N, H, W), dtype=torch.bool, device=device)       # (B * N, H, W)
#             src_embd = self.src_embed(src_mask)                                              # (B * N, C, H, W)
#
#         img_feat = img_feat.view(B * N, -1, H, W)                                            # (B * N, C, H, W)
#         img_prob = self.depth_net(img_feat).softmax(dim=1).unsqueeze(4)                      # (B * N, D, H, W, 1)
#         coords2d = (coords2d * img_prob).sum(dim=1).unsqueeze(4)                             # (B * N, H, W, 4, 1)
#
#         if ego2imgs.shape[-1] == 3:
#             ego2imgs = F.pad(ego2imgs, (0, 1, 0, 1), mode='constant', value=0.0)             # (B, N, 4, 4)
#             ego2imgs[..., -1, -1] = 1.0                                                      # (B, N, 4, 4)
#
#         img2egos = torch.inverse(ego2imgs)                                                   # (B, N, 4, 4)
#         img2egos = img2egos.view(B * N, 1, 1, 4, 4).repeat(1, H, W, 1, 1)                    # (B * N, H, W, 4, 4)
#         coords2d = torch.matmul(img2egos, coords2d).squeeze(-1)[..., :3]                     # (B * N, H, W, 3)
#         coords2d = coords2d.unsqueeze(1)                                                     # (B * N, 1, H, W, 3)
#
#         embeds3d = self.pos_embed.mean(dim=4).repeat(B, 1, 1, 1)                             # (B, C, X, Y)
#         embeds2d = self.position_encoder(coords2d)                                           # (B * N, C, H, W)
#         embeds2d = embeds2d + src_embd
#         if N > 1:
#             embeds2d = embeds2d.view(B, N, self.embed_dim, H, W)
#         return embeds2d, embeds3d


# class PositionEmbeddingProj(nn.Module):
#
#     def __init__(self,
#                  in_channels,
#                  embed_dim=256,
#                  x_bound=[5, 65, 0.234375],
#                  y_bound=[-15, 15, 0.234375],
#                  z_bound=[-1, 1, 0.2],
#                  d_bound=[3.0, 68.0, 0.2],
#                  normalize=True,
#                  LID=False,
#                  with_z=True):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.x_bound = x_bound
#         self.y_bound = y_bound
#         self.z_bound = z_bound
#         self.d_bound = d_bound
#         self.normalize = normalize
#         self.LID = LID
#         self.with_z = with_z
#
#         self.d_num = int((d_bound[1] - d_bound[0]) / d_bound[2])
#         self.x_num = int((x_bound[1] - x_bound[0]) / x_bound[2])
#         self.y_num = int((y_bound[1] - y_bound[0]) / y_bound[2])
#         self.z_num = int((z_bound[1] - z_bound[0]) / z_bound[2])
#
#         self.input_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)
#
#         self.position_encoder = nn.Sequential(
#             nn.Conv2d(self.d_num * 3, self.embed_dim * 2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(self.embed_dim * 2, self.embed_dim, kernel_size=1, stride=1, padding=0),
#         )
#
#     def forward(self, img_feat, ego2imgs=None, cam2imgs=None, pad_shape=None):
#         pad_h, pad_w = pad_shape[:2]
#         if len(img_feat.shape) == 4:
#             B, C, H, W = img_feat.shape
#             N = 1
#         else:
#             B, N, C, H, W = img_feat.shape
#         device = img_feat.device
#
#         coords_h = torch.arange(H, dtype=torch.float32, device=device) * pad_h / H
#         coords_w = torch.arange(W, dtype=torch.float32, device=device) * pad_w / W
#
#         if self.LID:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             index_1 = index + 1
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / (self.d_num * (1 + self.d_num))
#             coords_d = self.d_bound[0] + bin_size * index * index_1
#         else:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / self.d_num
#             coords_d = self.d_bound[0] + bin_size * index
#
#         D = coords_d.shape[0]
#         coords2d = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d]), dim=-1)        # (W, H, D, 3)
#         coords2d[..., :2] = coords2d[..., :2] * coords2d[..., 2:3]                            # (W, H, D, 3)
#
#         img2cams = torch.inverse(cam2imgs)                                                    # (B, N, 3, 3)
#         coord_2d = coords2d.view(1, 1, W, H, D, 3, 1).repeat(B, N, 1, 1, 1, 1, 1)             # (B, N, W, H, D, 3, 1)
#         img2cams = img2cams.view(B, N, 1, 1, 1, 3, 3).repeat(1, 1, W, H, D, 1, 1)             # (B, N, W, H, D, 3, 3)
#         coord_2d = torch.matmul(img2cams, coord_2d).squeeze(-1)[..., :3]                      # (B, N, W, H, D, 3)
#         coord_2d = coord_2d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, D * 3, H, W)   # (B * N, D * 3, H, W)
#         embeds2d = self.position_encoder(coord_2d)                                            # (B * N, C, H, W)
#
#         img_feat = img_feat.view(B * N, -1, H, W)                                             # (B * N, C, H, W)
#         img_feat = self.input_proj(img_feat)                                                  # (B * N, C, H, W)
#         return embeds2d, img_feat


# class PositionEmbeddingProjV2(nn.Module):
#
#     def __init__(self,
#                  in_channels,
#                  embed_dim=256,
#                  x_bound=[5, 65, 0.234375],
#                  y_bound=[-15, 15, 0.234375],
#                  z_bound=[-1, 1, 0.2],
#                  d_bound=[3.0, 68.0, 0.2],
#                  normalize=True,
#                  LID=False,
#                  with_z=True):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.x_bound = x_bound
#         self.y_bound = y_bound
#         self.z_bound = z_bound
#         self.d_bound = d_bound
#         self.normalize = normalize
#         self.LID = LID
#         self.with_z = with_z
#
#         self.d_num = int((d_bound[1] - d_bound[0]) / d_bound[2])
#         self.x_num = int((x_bound[1] - x_bound[0]) / x_bound[2])
#         self.y_num = int((y_bound[1] - y_bound[0]) / y_bound[2])
#         self.z_num = int((z_bound[1] - z_bound[0]) / z_bound[2])
#
#         self.input_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)
#
#         self.d_head = nn.Conv2d(self.embed_dim, self.d_num, kernel_size=1)
#
#         if self.with_z:
#             self.z_head = nn.Conv2d(self.embed_dim, self.z_num, kernel_size=1)
#             self.map_feat = nn.Parameter(torch.randn((1, self.embed_dim, self.x_num, self.y_num)))
#             nn.init.normal_(self.map_feat)
#
#         self.position_encoder_2d = nn.Sequential(
#             nn.Conv2d(self.d_num * 3, self.embed_dim * 2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(self.embed_dim * 2, self.embed_dim, kernel_size=1, stride=1, padding=0),
#         )
#
#         self.position_encoder_3d = nn.Sequential(
#             nn.Conv2d(3, self.embed_dim * 4, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(self.embed_dim * 4, self.embed_dim, kernel_size=1, stride=1, padding=0),
#         )
#
#     def forward(self, img_feat, ego2imgs=None, cam2imgs=None, pad_shape=None):
#         pad_h, pad_w = pad_shape[:2]
#         if len(img_feat.shape) == 4:
#             B, C, H, W = img_feat.shape
#             N = 1
#         else:
#             B, N, C, H, W = img_feat.shape
#         device = img_feat.device
#
#         coords_h = torch.arange(H, dtype=torch.float32, device=device) * pad_h / H
#         coords_w = torch.arange(W, dtype=torch.float32, device=device) * pad_w / W
#
#         if self.LID:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             index_1 = index + 1
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / (self.d_num * (1 + self.d_num))
#             coords_d = self.d_bound[0] + bin_size * index * index_1
#         else:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / self.d_num
#             coords_d = self.d_bound[0] + bin_size * index
#
#         D = coords_d.shape[0]
#         coords2d = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d]), dim=-1)        # (W, H, D, 3)
#         coords2d[..., :2] = coords2d[..., :2] * coords2d[..., 2:3]                            # (W, H, D, 3)
#
#         img2cams = torch.inverse(cam2imgs)                                                    # (B, N, 3, 3)
#         coord_2d = coords2d.view(1, 1, W, H, D, 3, 1).repeat(B, N, 1, 1, 1, 1, 1)             # (B, N, W, H, D, 3, 1)
#         img2cams = img2cams.view(B, N, 1, 1, 1, 3, 3).repeat(1, 1, W, H, D, 1, 1)             # (B, N, W, H, D, 3, 3)
#         coord_2d = torch.matmul(img2cams, coord_2d).squeeze(-1)[..., :3]                      # (B, N, W, H, D, 3)
#         coord_2d = coord_2d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, D * 3, H, W)   # (B * N, D * 3, H, W)
#         embeds2d = self.position_encoder_2d(coord_2d)                                         # (B * N, C, H, W)
#
#         if not self.with_z:
#             ego2imgs = F.pad(cam2imgs, (0, 1, 0, 1), mode='constant', value=0.0)              # (B, N, 4, 4)
#             ego2imgs[..., -1, -1] = 1.0
#
#         img2egos = torch.inverse(ego2imgs)                                                    # (B, N, 4, 4)
#         coords2d = torch.cat((coords2d, torch.ones_like(coords2d[..., :1])), dim=-1)          # (W, H, D, 4)
#         coords2d = coords2d.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)             # (B, N, W, H, D, 4, 1)
#         img2egos = img2egos.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)             # (B, N, W, H, D, 4, 4)
#         coords2d = torch.matmul(img2egos, coords2d).squeeze(-1)[..., :3]                      # (B, N, W, H, D, 3)
#         coords2d = coords2d.view(B * N, W, H, D, 3).permute(0, 3, 2, 1, 4)                    # (B * N, D, H, W, 3)
#
#         img_feat = img_feat.view(B * N, -1, H, W)                                             # (B * N, C, H, W)
#         img_feat = self.input_proj(img_feat)                                                  # (B * N, C, H, W)
#         img_feat = img_feat + embeds2d                                                        # (B * N, C, H, W)
#
#         img_prob = self.d_head(img_feat).softmax(dim=1).unsqueeze(4)                          # (B * N, D, H, W, 1)
#         coords2d = (coords2d * img_prob).sum(dim=1)                                           # (B * N, H, W, 3)
#         coords2d = coords2d.permute(0, 3, 1, 2).contiguous()                                  # (B * N, 3, H, W)
#
#         embeds2d = self.position_encoder_3d(coords2d)                                         # (B * N, C, H, W)
#         embeds2d = embeds2d.view(B, N, self.embed_dim, H, W)
#         if len(img_feat.shape) == 4:
#             embeds2d = embeds2d.squeeze(1)
#
#         if self.with_z:
#             coords_x = torch.arange(*self.x_bound, dtype=torch.float32, device=device)
#             coords_y = torch.arange(*self.y_bound, dtype=torch.float32, device=device)
#             coords_z = torch.arange(*self.z_bound, dtype=torch.float32, device=device)
#             coords3d = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z]), dim=-1)    # (X, Y, Z, 3)
#
#             X, Y, Z = coords3d.shape[:3]
#             coords3d = coords3d.view(1, X, Y, Z, 3).repeat(B, 1, 1, 1, 1)                     # (B, X, Y, Z, 3)
#             coords3d = coords3d.permute(0, 3, 1, 2, 4)                                        # (B, Z, X, Y, 3)
#
#             map_feat = self.map_feat.repeat(B, 1, 1, 1)                                       # (B, C, X, Y)
#             map_prob = self.z_head(map_feat).softmax(dim=1).unsqueeze(4)                      # (B, Z, X, Y, 1)
#             coords3d = (coords3d * map_prob).sum(dim=1)                                       # (B, X, Y, 3)
#             coords3d = coords3d.permute(0, 3, 1, 2).flip([2, 3]).contiguous()                 # (B, 3, X, Y)
#
#             embeds3d = self.position_encoder_3d(coords3d)
#             return embeds2d, embeds3d, img_feat, map_feat
#         else:
#             return embeds2d, img_feat


# class PositionEmbeddingProjV2(nn.Module):
#
#     def __init__(self,
#                  in_channels,
#                  embed_dim=256,
#                  x_bound=[5, 65, 0.234375],
#                  y_bound=[-15, 15, 0.234375],
#                  z_bound=[-1, 1, 0.2],
#                  d_bound=[3.0, 68.0, 0.2],
#                  normalize=True,
#                  LID=False,
#                  with_z=True,
#                  with_intrinsic=True):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.x_bound = x_bound
#         self.y_bound = y_bound
#         self.z_bound = z_bound
#         self.d_bound = d_bound
#         self.normalize = normalize
#         self.LID = LID
#         self.with_z = with_z
#         self.with_intrinsic = with_intrinsic
#
#         self.d_num = int((d_bound[1] - d_bound[0]) / d_bound[2])
#         self.x_num = int((x_bound[1] - x_bound[0]) / x_bound[2])
#         self.y_num = int((y_bound[1] - y_bound[0]) / y_bound[2])
#         self.z_num = int((z_bound[1] - z_bound[0]) / z_bound[2])
#
#         self.input_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)
#
#         self.d_head = nn.Conv2d(self.embed_dim, self.d_num, kernel_size=1)
#
#         if self.with_intrinsic:
#             self.intrinsic_encoder = nn.Sequential(
#                 nn.Conv2d(9, self.embed_dim * 4, kernel_size=1, stride=1, padding=0),
#                 nn.ReLU(),
#                 nn.Conv2d(self.embed_dim * 4, self.embed_dim, kernel_size=1, stride=1, padding=0),
#             )
#
#         if self.with_z:
#             self.z_head = nn.Conv2d(self.embed_dim, self.z_num, kernel_size=1)
#             self.map_feat = nn.Parameter(torch.randn((1, self.embed_dim, self.x_num, self.y_num)))
#             nn.init.normal_(self.map_feat)
#
#         self.position_encoder = nn.Sequential(
#             nn.Conv2d(3, self.embed_dim * 4, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(self.embed_dim * 4, self.embed_dim, kernel_size=1, stride=1, padding=0),
#         )
#
#     def forward(self, img_feat, ego2imgs, cam2imgs, pad_shape):
#         pad_h, pad_w = pad_shape[:2]
#         if len(img_feat.shape) == 4:
#             B, C, H, W = img_feat.shape
#             N = 1
#         else:
#             B, N, C, H, W = img_feat.shape
#         device = img_feat.device
#
#         coords_h = torch.arange(H, dtype=torch.float32, device=device) * pad_h / H
#         coords_w = torch.arange(W, dtype=torch.float32, device=device) * pad_w / W
#
#         if self.LID:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             index_1 = index + 1
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / (self.d_num * (1 + self.d_num))
#             coords_d = self.d_bound[0] + bin_size * index * index_1
#         else:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / self.d_num
#             coords_d = self.d_bound[0] + bin_size * index
#
#         D = coords_d.shape[0]
#         coords2d = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d]), dim=-1)  # (W, H, D, 3)
#         coords2d = torch.cat((coords2d, torch.ones_like(coords2d[..., :1])), dim=-1)
#         coords2d[..., :2] = coords2d[..., :2] * coords2d[..., 2:3]
#
#         # img2egos = []
#         # for img_meta in img_metas:
#         #     img2ego = []
#         #     if isinstance(img_meta['ego2img'], list):
#         #         for i in range(len(img_meta['ego2img'])):
#         #             img2ego.append(np.linalg.inv(img_meta['ego2img'][i]))
#         #     else:
#         #         img2ego.append(np.linalg.inv(img_meta['ego2img']))
#         #     img2egos.append(np.asarray(img2ego))
#         # img2egos = np.asarray(img2egos)
#         # img2egos = coords.new_tensor(img2egos)  # (B, N, 4, 4)
#
#         if ego2imgs.shape[-1] == 3:
#             ego2imgs = F.pad(ego2imgs, (0, 1, 0, 1), mode='constant', value=0.0)          # (B, N, 4, 4)
#             ego2imgs[..., -1, -1] = 1.0
#
#         img2egos = torch.inverse(ego2imgs)
#         coords2d = coords2d.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)         # (B, N, W, H, D, 4, 1)
#         img2egos = img2egos.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)         # (B, N, W, H, D, 4, 4)
#         coords2d = torch.matmul(img2egos, coords2d).squeeze(-1)[..., :3]                  # (B, N, W, H, D, 3)
#         coords2d = coords2d.view(B * N, W, H, D, 3).permute(0, 3, 2, 1, 4)                # (B * N, D, H, W, 3)
#
#         img_feat = img_feat.view(B * N, -1, H, W)                                         # (B * N, C, H, W)
#         img_feat = self.input_proj(img_feat)                                              # (B * N, C, H, W)
#
#         if self.with_intrinsic:
#             cam2imgs = cam2imgs.view(B * N, -1, 1, 1) / 100.0                             # (B * N, 9, 1, 1)
#             int_embd = self.intrinsic_encoder(cam2imgs)                                   # (B * N, C, 1, 1)
#             img_feat = img_feat + int_embd                                                # (B * N, C, H, W)
#
#         img_prob = self.d_head(img_feat).softmax(dim=1).unsqueeze(4)                      # (B * N, D, H, W, 1)
#         coords2d = (coords2d * img_prob).sum(dim=1)                                       # (B * N, H, W, 3)
#         coords2d = coords2d.permute(0, 3, 1, 2).contiguous()                              # (B * N, 3, H, W)
#
#         embeds2d = self.position_encoder(coords2d)                                        # (B * N, C, H, W)
#         embeds2d = embeds2d.view(B, N, self.embed_dim, H, W)
#         if len(img_feat.shape) == 4:
#             embeds2d = embeds2d.squeeze(1)
#
#         if self.with_z:
#             coords_x = torch.arange(*self.x_bound, dtype=torch.float32, device=device)
#             coords_y = torch.arange(*self.y_bound, dtype=torch.float32, device=device)
#             coords_z = torch.arange(*self.z_bound, dtype=torch.float32, device=device)
#             coords3d = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z]), dim=-1)    # (X, Y, Z, 3)
#
#             X, Y, Z = coords3d.shape[:3]
#             coords3d = coords3d.view(1, X, Y, Z, 3).repeat(B, 1, 1, 1, 1)                     # (B, X, Y, Z, 3)
#             coords3d = coords3d.permute(0, 3, 1, 2, 4)                                        # (B, Z, X, Y, 3)
#
#             map_feat = self.map_feat.repeat(B, 1, 1, 1)                                       # (B, C, X, Y)
#             map_prob = self.z_head(map_feat).softmax(dim=1).unsqueeze(4)                      # (B, Z, X, Y, 1)
#             coords3d = (coords3d * map_prob).sum(dim=1)                                       # (B, X, Y, 3)
#             coords3d = coords3d.permute(0, 3, 1, 2).flip([2, 3]).contiguous()                 # (B, 3, X, Y)
#
#             embeds3d = self.position_encoder(coords3d)
#             return embeds2d, embeds3d, img_feat, map_feat
#         else:
#             return embeds2d, img_feat


# class PositionEmbeddingProj(nn.Module):
#
#     def __init__(self,
#                  embed_dim=256,
#                  x_bound=[5, 65, 0.234375],
#                  y_bound=[-15, 15, 0.234375],
#                  z_bound=[-1, 1, 0.2],
#                  d_bound=[3.0, 68.0, 0.2],
#                  normalize=False,
#                  LID=False):
#         super().__init__()
#         self.d_num = int((d_bound[1] - d_bound[0]) / d_bound[2])
#         self.z_num = int((z_bound[1] - z_bound[0]) / z_bound[2])
#         self.embed_dim = embed_dim
#         self.x_bound = x_bound
#         self.y_bound = y_bound
#         self.z_bound = z_bound
#         self.d_bound = d_bound
#         self.normalize = normalize
#         self.LID = LID
#
#         self.position_encoder_2d = nn.Sequential(
#             nn.Conv2d(self.d_num * 3, self.embed_dim * 2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(self.embed_dim * 2, self.embed_dim, kernel_size=1, stride=1, padding=0),
#         )
#
#         self.position_encoder_3d = nn.Sequential(
#             nn.Conv2d(self.z_num * 3, self.embed_dim * 2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(self.embed_dim * 2, self.embed_dim, kernel_size=1, stride=1, padding=0),
#         )
#
#     def forward(self, img_feat, ego2imgs, pad_shape):
#         pad_h, pad_w = pad_shape[:2]
#         if len(img_feat.shape) == 4:
#             B, C, H, W = img_feat.shape
#             N = 1
#         else:
#             B, N, C, H, W = img_feat.shape
#         device = img_feat.device
#
#         coords_h = torch.arange(H, dtype=torch.float32, device=device) * pad_h / H
#         coords_w = torch.arange(W, dtype=torch.float32, device=device) * pad_w / W
#
#         if self.LID:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             index_1 = index + 1
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / (self.d_num * (1 + self.d_num))
#             coords_d = self.d_bound[0] + bin_size * index * index_1
#         else:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / self.d_num
#             coords_d = self.d_bound[0] + bin_size * index
#
#         D = coords_d.shape[0]
#         coords2d = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d]), dim=-1)  # (W, H, D, 3)
#         coords2d = torch.cat((coords2d, torch.ones_like(coords2d[..., :1])), dim=-1)
#         coords2d[..., :2] = coords2d[..., :2] * coords2d[..., 2:3]
#
#         # img2egos = []
#         # for img_meta in img_metas:
#         #     img2ego = []
#         #     if isinstance(img_meta['ego2img'], list):
#         #         for i in range(len(img_meta['ego2img'])):
#         #             img2ego.append(np.linalg.inv(img_meta['ego2img'][i]))
#         #     else:
#         #         img2ego.append(np.linalg.inv(img_meta['ego2img']))
#         #     img2egos.append(np.asarray(img2ego))
#         # img2egos = np.asarray(img2egos)
#         # img2egos = coords.new_tensor(img2egos)  # (B, N, 4, 4)
#
#         img2egos = torch.inverse(ego2imgs)
#         coords2d = coords2d.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)  # (B, N, W, H, D, 4, 1)
#         img2egos = img2egos.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)  # (B, N, W, H, D, 4, 4)
#         coords2d = torch.matmul(img2egos, coords2d).squeeze(-1)[..., :3]  # (B, N, W, H, D, 3)
#
#         if self.normalize:
#             coords2d[..., 0:1] = (coords2d[..., 0:1] - self.x_bound[0]) / (self.x_bound[1] - self.x_bound[0])
#             coords2d[..., 1:2] = (coords2d[..., 1:2] - self.y_bound[0]) / (self.y_bound[1] - self.y_bound[0])
#             coords2d[..., 2:3] = (coords2d[..., 2:3] - self.z_bound[0]) / (self.z_bound[1] - self.z_bound[0])
#
#         coords2d = coords2d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
#         embeds2d = self.position_encoder_2d(coords2d)
#         embeds2d = embeds2d.view(B, N, self.embed_dim, H, W)
#         if len(img_feat.shape) == 4:
#             embeds2d = embeds2d.squeeze(1)
#
#         coords_x = torch.arange(*self.x_bound, dtype=torch.float32, device=device)
#         coords_y = torch.arange(*self.y_bound, dtype=torch.float32, device=device)
#         coords_z = torch.arange(*self.z_bound, dtype=torch.float32, device=device)
#         coords3d = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z]), dim=-1)  # (X, Y, Z, 3)
#
#         X, Y, Z = coords3d.shape[:3]
#         coords3d = coords3d.view(1, X, Y, Z, 3).repeat(B, 1, 1, 1, 1)  # (B, X, Y, Z, 3)
#
#         if self.normalize:
#             coords3d[..., 0:1] = (coords3d[..., 0:1] - self.x_bound[0]) / (self.x_bound[1] - self.x_bound[0])
#             coords3d[..., 1:2] = (coords3d[..., 1:2] - self.y_bound[0]) / (self.y_bound[1] - self.y_bound[0])
#             coords3d[..., 2:3] = (coords3d[..., 2:3] - self.z_bound[0]) / (self.z_bound[1] - self.z_bound[0])
#
#         # coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
#         # coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
#         # coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
#         # coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
#         # coords3d = inverse_sigmoid(coords3d)
#
#         coords3d = coords3d.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, X, Y).flip([2, 3])
#         embeds3d = self.position_encoder_3d(coords3d)
#         return embeds2d, embeds3d


# class PositionEmbeddingProjV2(nn.Module):
#
#     def __init__(self,
#                  in_channels,
#                  embed_dim=256,
#                  x_bound=[5, 65, 0.234375],
#                  y_bound=[-15, 15, 0.234375],
#                  z_bound=[-1, 1, 0.2],
#                  d_bound=[3.0, 68.0, 0.2],
#                  normalize=True,
#                  LID=False):
#         super().__init__()
#         self.in_channels = in_channels
#         self.embed_dim = embed_dim
#         self.x_bound = x_bound
#         self.y_bound = y_bound
#         self.z_bound = z_bound
#         self.d_bound = d_bound
#         self.normalize = normalize
#         self.LID = LID
#
#         self.d_num = int((d_bound[1] - d_bound[0]) / d_bound[2])
#         self.x_num = int((x_bound[1] - x_bound[0]) / x_bound[2])
#         self.y_num = int((y_bound[1] - y_bound[0]) / y_bound[2])
#         self.z_num = int((z_bound[1] - z_bound[0]) / z_bound[2])
#
#         self.x_embed = nn.Embedding(self.x_num, self.embed_dim // 3)
#         self.y_embed = nn.Embedding(self.y_num, self.embed_dim // 3)
#         self.z_embed = nn.Embedding(self.z_num, self.embed_dim - self.embed_dim // 3 * 2)
#
#         self.d_head = nn.Conv2d(self.in_channels[0], self.d_num, kernel_size=1)
#         self.z_head = nn.Conv2d(self.in_channels[1], self.z_num, kernel_size=1)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.normal_(self.x_embed.weight)
#         nn.init.normal_(self.y_embed.weight)
#         nn.init.normal_(self.z_embed.weight)
#
#     def forward(self, img_feat, map_feat, ego2imgs, pad_shape):
#         pad_h, pad_w = pad_shape[:2]
#         if len(img_feat.shape) == 4:
#             B, C, H, W = img_feat.shape
#             N = 1
#         else:
#             B, N, C, H, W = img_feat.shape
#         device = img_feat.device
#
#         coords_h = torch.arange(H, dtype=torch.float32, device=device) * pad_h / H
#         coords_w = torch.arange(W, dtype=torch.float32, device=device) * pad_w / W
#
#         if self.LID:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             index_1 = index + 1
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / (self.d_num * (1 + self.d_num))
#             coords_d = self.d_bound[0] + bin_size * index * index_1
#         else:
#             index = torch.arange(start=0, end=self.d_num, step=1, dtype=torch.float32, device=device)
#             bin_size = (self.d_bound[1] - self.d_bound[0]) / self.d_num
#             coords_d = self.d_bound[0] + bin_size * index
#
#         D = coords_d.shape[0]
#         coords2d = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d]), dim=-1)  # (W, H, D, 3)
#         coords2d = torch.cat((coords2d, torch.ones_like(coords2d[..., :1])), dim=-1)
#         coords2d[..., :2] = coords2d[..., :2] * coords2d[..., 2:3]
#
#         # img2egos = []
#         # for img_meta in img_metas:
#         #     img2ego = []
#         #     if isinstance(img_meta['ego2img'], list):
#         #         for i in range(len(img_meta['ego2img'])):
#         #             img2ego.append(np.linalg.inv(img_meta['ego2img'][i]))
#         #     else:
#         #         img2ego.append(np.linalg.inv(img_meta['ego2img']))
#         #     img2egos.append(np.asarray(img2ego))
#         # img2egos = np.asarray(img2egos)
#         # img2egos = coords.new_tensor(img2egos)  # (B, N, 4, 4)
#
#         img2egos = torch.inverse(ego2imgs)
#         coords2d = coords2d.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)         # (B, N, W, H, D, 4, 1)
#         img2egos = img2egos.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)         # (B, N, W, H, D, 4, 4)
#         coords2d = torch.matmul(img2egos, coords2d).squeeze(-1)[..., :3]                  # (B, N, W, H, D, 3)
#
#         coords2d = coords2d.permute(0, 1, 4, 3, 2, 5).view(B * N, D, H, W, 3)             # (B * N, D, H, W, 3)
#         img_feat = img_feat.view(B * N, -1, H, W)                                         # (B * N, C, H, W)
#         img_prob = self.d_head(img_feat).softmax(dim=1).unsqueeze(4)                      # (B * N, D, H, W, 1)
#         coords2d = (coords2d * img_prob).sum(dim=1)                                       # (B * N, H, W, 3)
#
#         coords_x = (coords2d[..., 0:1] - self.x_bound[0]) / (self.x_bound[1] - self.x_bound[0]) * 2.0 - 1.0
#         coords_y = (coords2d[..., 1:2] - self.y_bound[0]) / (self.y_bound[1] - self.y_bound[0]) * 2.0 - 1.0
#         coords_z = (coords2d[..., 2:3] - self.z_bound[0]) / (self.z_bound[1] - self.z_bound[0]) * 2.0 - 1.0
#
#         coords_x = torch.cat([coords_x, torch.zeros_like(coords_x)], dim=-1)
#         coords_y = torch.cat([coords_y, torch.zeros_like(coords_y)], dim=-1)
#         coords_z = torch.cat([coords_z, torch.zeros_like(coords_z)], dim=-1)
#
#         x_embeds = self.x_embed.weight.view(1, self.x_num, 1, -1).repeat(B * N, 1, 1, 1).permute(0, 3, 1, 2)  # (B * N, C, X, 1)
#         y_embeds = self.y_embed.weight.view(1, self.y_num, 1, -1).repeat(B * N, 1, 1, 1).permute(0, 3, 1, 2)  # (B * N, C, Y, 1)
#         z_embeds = self.z_embed.weight.view(1, self.z_num, 1, -1).repeat(B * N, 1, 1, 1).permute(0, 3, 1, 2)  # (B * N, C, Z, 1)
#
#
#         embeds2d = self.position_encoder_2d(coords2d)
#         embeds2d = embeds2d.view(B, N, self.embed_dim, H, W)
#         if len(img_feat.shape) == 4:
#             embeds2d = embeds2d.squeeze(1)
#
#         coords_x = torch.arange(*self.x_bound, dtype=torch.float32, device=device)
#         coords_y = torch.arange(*self.y_bound, dtype=torch.float32, device=device)
#         coords_z = torch.arange(*self.z_bound, dtype=torch.float32, device=device)
#         coords3d = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z]), dim=-1)  # (X, Y, Z, 3)
#
#         X, Y, Z = coords3d.shape[:3]
#         coords3d = coords3d.view(1, X, Y, Z, 3).repeat(B, 1, 1, 1, 1)        # (B, X, Y, Z, 3)
#
#         if self.normalize:
#             coords3d[..., 0:1] = (coords3d[..., 0:1] - self.x_bound[0]) / (self.x_bound[1] - self.x_bound[0])
#             coords3d[..., 1:2] = (coords3d[..., 1:2] - self.y_bound[0]) / (self.y_bound[1] - self.y_bound[0])
#             coords3d[..., 2:3] = (coords3d[..., 2:3] - self.z_bound[0]) / (self.z_bound[1] - self.z_bound[0])
#
#         coords3d = coords3d.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, X, Y).flip([2, 3])
#         embeds3d = self.position_encoder_3d(coords3d)
#         return embeds2d, embeds3d
