"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn, Tensor
from typing import Optional
from .position_encoding import PositionEmbeddingSine
from .position_encoding import PositionEmbeddingLearnedV2 as PositionEmbeddingLearned


class SwapTransformer(nn.Module):

    def __init__(self, in_channels, src_shape=None, tgt_shape=(32, 32), d_model=512, n_heads=8,
                 num_memory_embeds=256, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False, return_intermediate_dec=False,
                 src_pos_encode='sine', tgt_pos_encode='learned', src_cam_encode=False,
                 tgt_cam_encode=False, use_fix_encode=False, use_checkpoint=False, use_input_proj=True):
        super().__init__()
        if use_input_proj:
            self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        else:
            self.input_proj = nn.Identity()

        encoder_layer = SwapTransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, activation,
                                                    normalize_before, use_checkpoint)
        encoder_norm = None  # nn.LayerNorm(d_model)
        self.encoder = SwapTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm,
                                              return_intermediate=return_intermediate_dec)

        self.reset_parameters()

        if src_pos_encode == 'sine':
            self.src_pos_embed = PositionEmbeddingSine(d_model, normalize=True)
        elif src_pos_encode == 'learned':
            self.src_pos_embed = PositionEmbeddingLearned(src_shape[-2:], d_model)
        else:
            self.src_pos_embed = None

        if tgt_pos_encode == 'sine':
            self.tgt_pos_embed = PositionEmbeddingSine(d_model, normalize=True)
        elif tgt_pos_encode == 'learned':
            self.tgt_pos_embed = PositionEmbeddingLearned(tgt_shape[-2:], d_model)
        else:
            self.tgt_pos_embed = None

        if use_fix_encode:
            if src_pos_encode == 'sine':
                src_mask = torch.zeros((1, *src_shape[-2:]), dtype=torch.bool)
                self.src_pos_embed = nn.Parameter(self.src_pos_embed(src_mask), requires_grad=False)
            if tgt_pos_encode == 'sine':
                tgt_mask = torch.zeros((1, *tgt_shape[-2:]), dtype=torch.bool)
                self.tgt_pos_embed = nn.Parameter(self.tgt_pos_embed(tgt_mask), requires_grad=False)

        if src_cam_encode:
            self.src_cam_embed = nn.Embedding(src_shape[0], d_model)
        else:
            self.src_cam_embed = None
        if tgt_cam_encode:
            self.tgt_cam_embed = nn.Embedding(tgt_shape[0], d_model)
        else:
            self.tgt_cam_embed = None

        self.mem_pos_embed = nn.Embedding(num_memory_embeds, d_model)

        self.src_shape = src_shape
        self.tgt_shape = tgt_shape
        self.src_pos_encode = src_pos_encode
        self.tgt_pos_encode = tgt_pos_encode
        self.src_cam_encode = src_cam_encode
        self.tgt_cam_encode = tgt_cam_encode
        self.use_fix_encode = use_fix_encode

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                src,
                tgt=None,
                src_mask=None,
                tgt_mask=None,
                src_pos_embed=None,
                tgt_pos_embed=None):
        if isinstance(src, (list, tuple)):
            src = src[-1]                                                                            # (N, C, H, W)

        if isinstance(src_mask, (list, tuple)):
            src_mask = src_mask[-1]                                                                  # (N, H, W)

        if len(src.shape) == 4:
            bs, _, h_src, w_src = src.shape
            d_src = None
            src = self.input_proj(src)                                                               # (N, C, H, W)
        else:
            bs, d_src, _, h_src, w_src = src.shape
            src = self.input_proj(src.flatten(0, 1))
            src = src.view(bs, d_src, -1, h_src, w_src)                                              # (N, D, C, H, W)

        if self.use_fix_encode:
            src_mask = None
            if src_pos_embed is None:
                if d_src is None:
                    if self.src_pos_encode == 'sine':
                        src_pos_embed = self.src_pos_embed                                           # (1, C, H, W)
                        src_pos_embed = src_pos_embed.repeat(bs, 1, 1, 1)                            # (N, C, H, W)
                    else:
                        src_pos_embed = self.src_pos_embed.pos_embed.weight                          # (H * W, C)
                        src_pos_embed = src_pos_embed.view(1, h_src, w_src, -1)                      # (1, H, W, C)
                        src_pos_embed = src_pos_embed.permute(0, 3, 1, 2).repeat(bs, 1, 1, 1)        # (N, C, H, W)
                else:
                    if self.src_pos_encode == 'sine':
                        src_pos_embed = self.src_pos_embed.unsqueeze(1)                              # (1, 1, C, H, W)
                        src_pos_embed = src_pos_embed.repeat(bs, d_src, 1, 1, 1)                     # (N, D, C, H, W)
                    else:
                        src_pos_embed = self.src_pos_embed.pos_embed.weight                          # (H * W, C)
                        src_pos_embed = src_pos_embed.view(1, 1, h_src, w_src, -1)                   # (1, 1, H, W, C)
                        src_pos_embed = src_pos_embed.permute(0, 1, 3, 1, 2).repeat(bs, d_src, 1, 1, 1)      # (N, D, C, H, W)
        else:
            if src_mask is None:
                if d_src is None:
                    src_mask = torch.zeros((bs, h_src, w_src), dtype=torch.bool, device=src.device)  # (N, H, W)
                else:
                    src_mask = torch.zeros((bs, d_src, h_src, w_src), dtype=torch.bool, device=src.device)   # (N, D, H, W)
            if src_pos_embed is None:
                if d_src is None:
                    src_pos_embed = self.src_pos_embed(src_mask)                                     # (N, C, H, W)
                else:
                    src_pos_embed = self.src_pos_embed(src_mask.flatten(0, 1))                       # (N * D, C, H, W)
                    src_pos_embed = src_pos_embed.view(bs, d_src, -1, h_src, w_src)                  # (N, D, C, H, W)

        # For source
        if self.src_cam_encode:
            src_pos_embed = src_pos_embed + self.src_cam_embed.weight[:, :, None, None]              # (N, D, C, H, W)

        if d_src is not None:
            src_pos_embed = src_pos_embed.transpose(1, 2)                                            # (N, C, D, H, W)
            src = src.transpose(1, 2)                                                                # (N, C, D, H, W)

        # src = src + src_pos_embed                                                                  # (N, C, H, W)

        src = src.flatten(2).permute(2, 0, 1)                                                        # (H * W, N, C)
        src_mask = src_mask.flatten(1) if src_mask is not None else src_mask                         # (N, H * W)
        src_pos_embed = src_pos_embed.flatten(2).permute(2, 0, 1)                                    # (H * W, N, C)

        # For target
        if tgt is not None:
            h_tgt, w_tgt = tgt.shape[-2:]
        elif tgt_pos_embed is not None:
            h_tgt, w_tgt = tgt_pos_embed.shape[-2:]
        else:
            h_tgt, w_tgt = self.tgt_shape[-2:]

        if self.use_fix_encode:
            tgt_mask = None
            if tgt_pos_embed is None:
                if self.tgt_pos_encode == 'sine':
                    tgt_pos_embed = self.tgt_pos_embed                                               # (1, C, H, W)
                    tgt_pos_embed = tgt_pos_embed.repeat(bs, 1, 1, 1)                                # (N, C, H, W)
                else:
                    tgt_pos_embed = self.tgt_pos_embed.pos_embed.weight                              # (H * W, C)
                    tgt_pos_embed = tgt_pos_embed.view(1, h_tgt, w_tgt, -1)                          # (1, H, W, C)
                    tgt_pos_embed = tgt_pos_embed.permute(0, 3, 1, 2).repeat(bs, 1, 1, 1)            # (N, C, H, W)
        else:
            if tgt_mask is None:
                tgt_mask = torch.zeros((bs, h_tgt, w_tgt), dtype=torch.bool, device=src.device)      # (N, H, W)
            if tgt_pos_embed is None:
                tgt_pos_embed = self.tgt_pos_embed(tgt_mask)                                         # (N, C, H, W)

        if self.tgt_cam_encode:
            tgt_pos_embed = tgt_pos_embed + self.tgt_cam_embed.weight[:, :, None, None]              # (N, C, H, W)

        if tgt is None:
            tgt = tgt_pos_embed                                                                      # (N, C, H, W)

        # else:
        #     tgt = tgt + tgt_pos_embed                                                              # (N, C, H, W)

        tgt = tgt.flatten(2).permute(2, 0, 1)                                                        # (H * W, N, C)
        tgt_mask = tgt_mask.flatten(1) if tgt_mask is not None else tgt_mask                         # (N, H * W)
        tgt_pos_embed = tgt_pos_embed.flatten(2).permute(2, 0, 1)                                    # (H * W, N, C)

        # For memory
        mem_pos_embed = self.mem_pos_embed.weight                                                    # (L, C)
        mem_pos_embed = mem_pos_embed.unsqueeze(dim=1).repeat(1, bs, 1)                              # (L, N, C)
        mem = mem_pos_embed                                                                          # (L, N, C)
        mem_mask = None

        com = torch.cat((src, tgt), dim=0)                                                           # (H * W, N, C)
        com_mask = torch.cat((src_mask, tgt_mask), dim=1) if not self.use_fix_encode else None       # (N, H * W)
        com_pos_embed = torch.cat((src_pos_embed, tgt_pos_embed), dim=0)                             # (H * W, N, C)

        com_list, mem_list = self.encoder(src=com,
                                          tgt=mem,
                                          src_key_padding_mask=com_mask,
                                          tgt_key_padding_mask=mem_mask,
                                          src_pos_embed=com_pos_embed,
                                          tgt_pos_embed=mem_pos_embed)                               # (H * W, N, C)

        src_list = []
        tgt_list = []
        for x in com_list:
            x_src, x_tgt = x.split([src.size(0), tgt.size(0)], dim=0)
            if d_src is None:
                x_src = x_src.view(h_src, w_src, bs, -1).permute(2, 3, 0, 1).contiguous()            # (N, C, H, W)
            else:
                x_src = x_src.view(d_src, h_src, w_src, bs, -1).permute(3, 0, 4, 1, 2).contiguous()  # (N, D, C, H, W)

            x_tgt = x_tgt.view(h_tgt, w_tgt, bs, -1).permute(2, 3, 0, 1).contiguous()                # (N, C, H, W)
            src_list.append(x_src)
            tgt_list.append(x_tgt)
        return src_list, tgt_list, mem_list


class SwapTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                src,
                tgt,
                src_attn_mask: Optional[Tensor] = None,
                tgt_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos_embed: Optional[Tensor] = None,
                tgt_pos_embed: Optional[Tensor] = None):

        src_list = []
        tgt_list = []
        for layer in self.layers:
            src, tgt = layer(src,
                             tgt,
                             src_attn_mask,
                             tgt_attn_mask,
                             src_key_padding_mask,
                             tgt_key_padding_mask,
                             src_pos_embed,
                             tgt_pos_embed)
            src_list.append(src)
            tgt_list.append(tgt)
        return src_list, tgt_list


class SwapTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False, use_checkpoint=False):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout6 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

        self.normalize_before = normalize_before
        self.use_checkpoint = use_checkpoint

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _forward(self,
                 src,
                 tgt,
                 src_attn_mask: Optional[Tensor] = None,
                 tgt_attn_mask: Optional[Tensor] = None,
                 src_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 src_pos_embed: Optional[Tensor] = None,
                 tgt_pos_embed: Optional[Tensor] = None):
        # src ---> tgt
        q = self.with_pos_embed(tgt, tgt_pos_embed)
        k = self.with_pos_embed(src, src_pos_embed)
        v = src

        tgt2 = self.attn1(query=q, key=k, value=v, attn_mask=src_attn_mask,
                          key_padding_mask=src_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout2(self.activation1(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt)

        # tgt ---> src
        q = k
        k = self.with_pos_embed(tgt, tgt_pos_embed)
        v = tgt

        src2 = self.attn2(query=q, key=k, value=v, attn_mask=tgt_attn_mask,
                          key_padding_mask=tgt_key_padding_mask)[0]
        src = src + self.dropout4(src2)
        src = self.norm3(src)

        src2 = self.linear4(self.dropout5(self.activation2(self.linear3(src))))
        src = src + self.dropout6(src2)
        src = self.norm4(src)
        return src, tgt

    def forward(self,
                src,
                tgt,
                src_attn_mask: Optional[Tensor] = None,
                tgt_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos_embed: Optional[Tensor] = None,
                tgt_pos_embed: Optional[Tensor] = None):
        if self.use_checkpoint and self.training:
            out = checkpoint.checkpoint(self._forward,
                                        src,
                                        tgt,
                                        src_attn_mask,
                                        tgt_attn_mask,
                                        src_key_padding_mask,
                                        tgt_key_padding_mask,
                                        src_pos_embed,
                                        tgt_pos_embed)
        else:
            out = self._forward(src,
                                tgt,
                                src_attn_mask,
                                tgt_attn_mask,
                                src_key_padding_mask,
                                tgt_key_padding_mask,
                                src_pos_embed,
                                tgt_pos_embed)
        return out


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# attn1_src, attn1_tgt = attn1.split([src.size(0), tgt.size(0)], dim=2)
# attn1_src = attn1_src.view(bs, -1, h, w)[:, :20]
# attn1_tgt = attn1_tgt.view(bs, -1, *tgt_shape)[:, :20]
# print(attn1_src.shape)
# print(attn1_tgt.shape)
#
# attn2_src, attn2_tgt = attn2.split([src.size(0), tgt.size(0)], dim=1)
# attn2_src = attn2_src.permute(0, 2, 1).contiguous().view(bs, -1, h, w)[:, :20]
# attn2_tgt = attn2_tgt.permute(0, 2, 1).contiguous().view(bs, -1, *tgt_shape)[:, :20]
#
# import os
# import cv2
# import shutil
# import numpy as np
#
# shutil.rmtree('/data/sets/czy/')
#
# for i, attn_img in enumerate(attn1_src):
#     for j, attn_q in enumerate(attn_img):
#         os.makedirs('/data/sets/czy/query_src_{}'.format(j), exist_ok=True)
#         attn_q = attn_q.cpu().numpy()
#         attn_q = (attn_q - attn_q.min()) / (attn_q.max() - attn_q.min())
#         attn_q = (attn_q * 255).astype(np.uint8)
#         cv2.imwrite(os.path.join('/data/sets/czy/query_src_{}'.format(j), '{}.png'.format(i)), attn_q)
#
# for i, attn_img in enumerate(attn1_tgt):
#     for j, attn_q in enumerate(attn_img):
#         os.makedirs('/data/sets/czy/query_tgt_{}'.format(j), exist_ok=True)
#         # attn_m = float(attn_q.min().cpu().numpy())
#         # attn_q = attn_q.clamp(min=attn_m * 5)
#         attn_q = attn_q.cpu().numpy()
#         attn_q = (attn_q - attn_q.min()) / (attn_q.max() - attn_q.min())
#         attn_q = (attn_q * 255).astype(np.uint8)
#         cv2.imwrite(os.path.join('/data/sets/czy/query_tgt_{}'.format(j), '{}.png'.format(i)), attn_q)
#
# import sys
# sys.exit()
