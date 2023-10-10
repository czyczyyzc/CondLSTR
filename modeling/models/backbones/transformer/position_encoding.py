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
