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


class Transformer(nn.Module):

    def __init__(self, in_channels, src_shape=None, tgt_shape=(32, 32), d_model=512, n_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False, return_intermediate_dec=False,
                 src_pos_encode='sine', tgt_pos_encode='learned', src_cam_encode=False,
                 tgt_cam_encode=False, use_fix_encode=False, use_checkpoint=False, use_input_proj=True,
                 src_down_scale=None, tgt_down_scale=None, mem_down_scale=None):
        super().__init__()
        if use_input_proj:
            self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        else:
            self.input_proj = nn.Identity()

        encoder_layer = TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, activation,
                                                normalize_before, use_checkpoint, src_down_scale)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout, activation,
                                                normalize_before, use_checkpoint, tgt_down_scale)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self.dropout = nn.Dropout(dropout)

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

        self.src_shape = src_shape
        self.tgt_shape = tgt_shape
        self.src_pos_encode = src_pos_encode
        self.tgt_pos_encode = tgt_pos_encode
        self.src_cam_encode = src_cam_encode
        self.tgt_cam_encode = tgt_cam_encode
        self.use_fix_encode = use_fix_encode
        self.src_down_scale = src_down_scale
        self.tgt_down_scale = tgt_down_scale
        self.mem_down_scale = mem_down_scale

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def share_encoder(self):
        del self.input_proj
        del self.encoder
        del self.src_pos_embed
        del self.src_cam_embed

    def forward(self, src, src_mask=None):
        encoder_output = self.forward_encoder(src, src_mask)
        decoder_output = self.forward_decoder(encoder_output)
        return decoder_output

    def forward_encoder(self, src, src_mask=None, src_pos_embed=None):
        if isinstance(src, (list, tuple)):
            src = src[-1]                                                                            # (N, C, H, W)

        if isinstance(src_mask, (list, tuple)):
            src_mask = src_mask[-1]                                                                  # (N, H, W)

        if len(src.shape) == 4:
            bs, _, h, w = src.shape
            d = 1
            src = self.input_proj(src)                                                               # (N, C, H, W)
        else:
            bs, d, _, h, w = src.shape
            src = self.input_proj(src.flatten(0, 1))
            src = src.view(bs, d, -1, h, w)

        if self.src_cam_encode:
            if len(src.shape) == 4:
                src = src.unsqueeze(1)                                                               # (N, D, C, H, W)
                if src_mask is not None:
                    src_mask = src_mask.unsqueeze(1)                                                 # (N, D, H, W)
                if src_pos_embed is not None:
                    src_pos_embed = src_pos_embed.unsqueeze(1)                                       # (N, D, C, H, W)

            if self.use_fix_encode:
                src_mask = None
                if src_pos_embed is None:
                    if self.src_pos_encode == 'sine':
                        src_pos_embed = self.src_pos_embed.unsqueeze(1)                              # (1, 1, C, H, W)
                        src_pos_embed = src_pos_embed.repeat(bs, d, 1, 1, 1)                         # (N, D, C, H, W)
                    else:
                        src_pos_embed = self.src_pos_embed.pos_embed.weight                          # (H * W, C)
                        src_pos_embed = src_pos_embed.view(1, 1, *self.src_shape[-2:], -1)           # (1, 1, H, W, C)
                        src_pos_embed = src_pos_embed.permute(0, 1, 3, 1, 2).repeat(bs, d, 1, 1, 1)  # (N, D, C, H, W)
            else:
                if src_pos_embed is None:
                    if src_mask is None:
                        src_mask = torch.zeros((bs, d, h, w), dtype=torch.bool, device=src.device)   # (N, D, H, W)
                    src_pos_embed = self.src_pos_embed(src_mask.flatten(0, 1))                       # (N * D, C, H, W)
                    src_pos_embed = src_pos_embed.view(bs, d, -1, h, w)                              # (N, D, C, H, W)

            src_pos_embed = src_pos_embed + self.src_cam_embed.weight[:, :, None, None]              # (N, D, C, H, W)
            src_pos_embed = src_pos_embed.transpose(1, 2)                                            # (N, C, D, H, W)
            src = src.transpose(1, 2)                                                                # (N, C, D, H, W)
        else:
            if self.use_fix_encode:
                src_mask = None
                if src_pos_embed is None:
                    if self.src_pos_encode == 'sine':
                        src_pos_embed = self.src_pos_embed                                           # (1, C, H, W)
                        src_pos_embed = src_pos_embed.repeat(bs, 1, 1, 1)                            # (N, C, H, W)
                    else:
                        src_pos_embed = self.src_pos_embed.pos_embed.weight                          # (H * W, C)
                        src_pos_embed = src_pos_embed.view(1, *self.src_shape[-2:], -1)              # (1, H, W, C)
                        src_pos_embed = src_pos_embed.permute(0, 3, 1, 2).repeat(bs, 1, 1, 1)        # (N, C, H, W)
            else:
                if src_pos_embed is None:
                    if src_mask is None:
                        src_mask = torch.zeros((bs, h, w), dtype=torch.bool, device=src.device)      # (N, H, W)
                    src_pos_embed = self.src_pos_embed(src_mask)                                     # (N, C, H, W)

        # src = src + src_pos_embed                                                                  # (N, C, H, W)
        #
        # if self.src_down_scale is not None:
        #     if src_mask is not None:
        #         src_mask = F.interpolate(src_mask[:, None].float(), scale_factor=1./self.src_down_scale,
        #                                  mode='nearest')[:, 0].bool()

        src_shape = src_pos_embed.shape[-2:]
        src = src.flatten(2).permute(2, 0, 1)                                                        # (H * W, N, C)
        src_mask = src_mask.flatten(1) if src_mask is not None else src_mask                         # (N, H * W)
        src_pos_embed = src_pos_embed.flatten(2).permute(2, 0, 1)                                    # (H * W, N, C)

        memory = self.encoder(src, src_key_padding_mask=src_mask,
                              src_pos_embed=src_pos_embed, src_shape=src_shape)                      # (H * W, N, C)
        return memory, src_mask, src_pos_embed, src_shape

    def forward_decoder(self, encoder_output, tgt=None, tgt_mask=None, tgt_pos_embed=None):
        memory, src_mask, src_pos_embed, src_shape = encoder_output
        bs, c = memory.shape[-2:]

        if self.mem_down_scale is not None:
            h, w = src_shape
            memory = memory.view(h, w, bs, -1).permute(2, 3, 0, 1).contiguous()                      # (N, C, H, W)
            memory = F.interpolate(memory, scale_factor=1./self.mem_down_scale, mode='nearest')      # (N, C, H, W)
            memory = memory.view(bs, c, -1).permute(2, 0, 1).contiguous()                            # (H * W, N, C)
            src_pos_embed = src_pos_embed.view(h, w, bs, -1).permute(2, 3, 0, 1).contiguous()        # (N, C, H, W)
            src_pos_embed = F.interpolate(src_pos_embed, scale_factor=1./self.mem_down_scale, mode='nearest')  # (N, C, H, W)
            src_pos_embed = src_pos_embed.view(bs, c, -1).permute(2, 0, 1).contiguous()              # (H * W, N, C)
            if src_mask is not None:
                src_mask = src_mask.view(bs, h, w)[:, None].float()                                      # (N, 1, H, W)
                src_mask = F.interpolate(src_mask, scale_factor=1./self.mem_down_scale, mode='nearest')  # (N, 1, H, W)
                src_mask = src_mask[:, 0].bool().flatten(1)                                              # (N, H * W)

        if self.use_fix_encode:
            tgt_mask = None
            if tgt_pos_embed is None:
                if self.tgt_pos_encode == 'sine':
                    tgt_pos_embed = self.tgt_pos_embed                                               # (1, C, H, W)
                    tgt_pos_embed = tgt_pos_embed.repeat(bs, 1, 1, 1)                                # (N, C, H, W)
                else:
                    tgt_pos_embed = self.tgt_pos_embed.pos_embed.weight                              # (H * W, C)
                    tgt_pos_embed = tgt_pos_embed.view(1, *self.tgt_shape[-2:], -1)                  # (1, H, W, C)
                    tgt_pos_embed = tgt_pos_embed.permute(0, 3, 1, 2).repeat(bs, 1, 1, 1)            # (N, C, H, W)
        else:
            if tgt_pos_embed is None:
                if tgt_mask is None:
                    tgt_mask = torch.zeros((bs, *self.tgt_shape[-2:]),
                                           dtype=torch.bool, device=memory.device)                   # (N, H, W)
                tgt_pos_embed = self.tgt_pos_embed(tgt_mask)                                         # (N, C, H, W)

        if self.tgt_cam_encode:
            tgt_pos_embed = tgt_pos_embed + self.tgt_cam_embed.weight[:, :, None, None]              # (N, C, H, W)

        if tgt is None:
            tgt = tgt_pos_embed                                                                      # (N, C, H, W)

        # else:
        #     tgt = tgt + tgt_pos_embed                                                              # (N, C, H, W)
        #
        # if self.tgt_down_scale is not None:
        #     if tgt_mask is not None:
        #         tgt_mask = F.interpolate(tgt_mask[:, None].float(), scale_factor=1./self.tgt_down_scale,
        #                                  mode='nearest')[:, 0].bool()

        tgt_shape = tgt_pos_embed.shape[-2:]
        tgt = tgt.flatten(2).permute(2, 0, 1)                                                        # (H * W, N, C)
        tgt_mask = tgt_mask.flatten(1) if tgt_mask is not None else tgt_mask                         # (N, H * W)
        tgt_pos_embed = tgt_pos_embed.flatten(2).permute(2, 0, 1)                                    # (H * W, N, C)

        hs, attn = self.decoder(tgt, memory, memory_key_padding_mask=src_mask,
                                tgt_pos_embed=tgt_pos_embed, memory_pos_embed=src_pos_embed,
                                tgt_shape=tgt_shape)                                                 # (M, H * W, N, C)
        ys = [x.view(*tgt_shape, bs, -1).permute(2, 3, 0, 1).contiguous() for x in hs]
        return ys, hs, tgt_mask, tgt_pos_embed, attn


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_pos_embed: Optional[Tensor] = None,
                src_shape: Optional[tuple] = None):
        output = src

        for layer in self.layers:
            output = layer(output,
                           src_attn_mask=src_attn_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           src_pos_embed=src_pos_embed,
                           src_shape=src_shape)

        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                memory,
                tgt_attn_mask: Optional[Tensor] = None,
                memory_attn_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_pos_embed: Optional[Tensor] = None,
                memory_pos_embed: Optional[Tensor] = None,
                tgt_shape: Optional[tuple] = None):
        output = tgt

        intermediate = []
        attn = None
        for layer in self.layers:
            output, attn = layer(output,
                                 memory,
                                 tgt_attn_mask=tgt_attn_mask,
                                 memory_attn_mask=memory_attn_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask,
                                 tgt_pos_embed=tgt_pos_embed,
                                 memory_pos_embed=memory_pos_embed,
                                 tgt_shape=tgt_shape)
            intermediate.append(self.norm(output))
        return intermediate, attn


# class TransformerDecoder(nn.Module):
#
#     def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#         self.return_intermediate = return_intermediate
#
#     def forward(self,
#                 tgt,
#                 memory,
#                 tgt_attn_mask: Optional[Tensor] = None,
#                 memory_attn_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None,
#                 tgt_pos_embed: Optional[Tensor] = None,
#                 memory_pos_embed: Optional[Tensor] = None):
#         output = tgt
#
#         intermediate = []
#
#         for layer in self.layers:
#             output = layer(output,
#                            memory,
#                            tgt_attn_mask=tgt_attn_mask,
#                            memory_attn_mask=memory_attn_mask,
#                            tgt_key_padding_mask=tgt_key_padding_mask,
#                            memory_key_padding_mask=memory_key_padding_mask,
#                            tgt_pos_embed=tgt_pos_embed,
#                            memory_pos_embed=memory_pos_embed)
#             if self.return_intermediate:
#                 intermediate.append(self.norm(output))
#
#         if self.norm is not None:
#             output = self.norm(output)
#             if self.return_intermediate:
#                 intermediate.pop()
#                 intermediate.append(output)
#
#         if self.return_intermediate:
#             return torch.stack(intermediate)
#
#         return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False, use_checkpoint=False, src_down_scale=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.use_checkpoint = use_checkpoint
        self.src_down_scale = src_down_scale

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_attn_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     src_pos_embed: Optional[Tensor] = None,
                     src_shape: Optional[tuple] = None):
        q = k = self.with_pos_embed(src, src_pos_embed)
        v = src
        h, w = src_shape

        if self.src_down_scale is not None:
            k = k.view(h, w, k.size(1), -1).permute(2, 3, 0, 1).contiguous()                 # (N, C, H, W)
            v = v.view(h, w, v.size(1), -1).permute(2, 3, 0, 1).contiguous()                 # (N, C, H, W)
            k = F.interpolate(k, scale_factor=1./self.src_down_scale, mode='nearest')        # (N, C, H, W)
            v = F.interpolate(v, scale_factor=1./self.src_down_scale, mode='nearest')        # (N, C, H, W)
            k = k.view(k.size(0), k.size(1), -1).permute(2, 0, 1).contiguous()               # (H * W, N, C)
            v = v.view(v.size(0), v.size(1), -1).permute(2, 0, 1).contiguous()               # (H * W, N, C)
            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask.view(-1, h, w)[:, None].float()  # (N, 1, H, W)
                src_key_padding_mask = F.interpolate(src_key_padding_mask, scale_factor=1./self.src_down_scale, mode='nearest')  # (N, 1, H, W)
                src_key_padding_mask = src_key_padding_mask[:, 0].bool().flatten(1)          # (N, H * W)

        src2 = self.self_attn(q, k, value=v, attn_mask=src_attn_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self,
                    src,
                    src_attn_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    src_pos_embed: Optional[Tensor] = None,
                    src_shape: Optional[tuple] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, src_pos_embed)
        v = src2
        h, w = src_shape

        if self.src_down_scale is not None:
            k = k.view(h, w, k.size(1), -1).permute(2, 3, 0, 1).contiguous()                 # (N, C, H, W)
            v = v.view(h, w, v.size(1), -1).permute(2, 3, 0, 1).contiguous()                 # (N, C, H, W)
            k = F.interpolate(k, scale_factor=1./self.src_down_scale, mode='nearest')        # (N, C, H, W)
            v = F.interpolate(v, scale_factor=1./self.src_down_scale, mode='nearest')        # (N, C, H, W)
            k = k.view(k.size(0), k.size(1), -1).permute(2, 0, 1).contiguous()               # (H * W, N, C)
            v = v.view(v.size(0), v.size(1), -1).permute(2, 0, 1).contiguous()               # (H * W, N, C)
            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask.view(-1, h, w)[:, None].float()  # (N, 1, H, W)
                src_key_padding_mask = F.interpolate(src_key_padding_mask, scale_factor=1./self.src_down_scale, mode='nearest')  # (N, 1, H, W)
                src_key_padding_mask = src_key_padding_mask[:, 0].bool().flatten(1)          # (N, H * W)

        src2 = self.self_attn(q, k, value=v, attn_mask=src_attn_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def _forward(self,
                 src,
                 src_attn_mask: Optional[Tensor] = None,
                 src_key_padding_mask: Optional[Tensor] = None,
                 src_pos_embed: Optional[Tensor] = None,
                 src_shape: Optional[tuple] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_attn_mask, src_key_padding_mask, src_pos_embed, src_shape)
        return self.forward_post(src, src_attn_mask, src_key_padding_mask, src_pos_embed, src_shape)

    def forward(self,
                src,
                src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_pos_embed: Optional[Tensor] = None,
                src_shape: Optional[tuple] = None):
        if self.use_checkpoint and self.training:
            out = checkpoint.checkpoint(self._forward,
                                        src,
                                        src_attn_mask,
                                        src_key_padding_mask,
                                        src_pos_embed,
                                        src_shape)
        else:
            out = self._forward(src,
                                src_attn_mask,
                                src_key_padding_mask,
                                src_pos_embed,
                                src_shape)
        return out


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False, use_checkpoint=False, tgt_down_scale=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.use_checkpoint = use_checkpoint
        self.tgt_down_scale = tgt_down_scale

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     tgt_attn_mask: Optional[Tensor] = None,
                     memory_attn_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     tgt_pos_embed: Optional[Tensor] = None,
                     memory_pos_embed: Optional[Tensor] = None,
                     tgt_shape: Optional[tuple] = None):
        q = k = self.with_pos_embed(tgt, tgt_pos_embed)
        v = tgt
        h, w = tgt_shape

        if self.tgt_down_scale is not None:
            k = k.view(h, w, k.size(1), -1).permute(2, 3, 0, 1).contiguous()                 # (N, C, H, W)
            v = v.view(h, w, v.size(1), -1).permute(2, 3, 0, 1).contiguous()                 # (N, C, H, W)
            k = F.interpolate(k, scale_factor=1./self.tgt_down_scale, mode='nearest')        # (N, C, H, W)
            v = F.interpolate(v, scale_factor=1./self.tgt_down_scale, mode='nearest')        # (N, C, H, W)
            k = k.view(k.size(0), k.size(1), -1).permute(2, 0, 1).contiguous()               # (H * W, N, C)
            v = v.view(v.size(0), v.size(1), -1).permute(2, 0, 1).contiguous()               # (H * W, N, C)
            if tgt_key_padding_mask is not None:
                tgt_key_padding_mask = tgt_key_padding_mask.view(-1, h, w)[:, None].float()  # (N, 1, H, W)
                tgt_key_padding_mask = F.interpolate(tgt_key_padding_mask, scale_factor=1./self.tgt_down_scale, mode='nearest')  # (N, 1, H, W)
                tgt_key_padding_mask = tgt_key_padding_mask[:, 0].bool().flatten(1)          # (N, H * W)

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_attn_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos_embed),
        #                            key=self.with_pos_embed(memory, memory_pos_embed),
        #                            value=memory, attn_mask=memory_attn_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]

        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos_embed),
                                         key=self.with_pos_embed(memory, memory_pos_embed),
                                         value=memory, attn_mask=memory_attn_mask,
                                         key_padding_mask=memory_key_padding_mask)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn

    def forward_pre(self,
                    tgt,
                    memory,
                    tgt_attn_mask: Optional[Tensor] = None,
                    memory_attn_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    tgt_pos_embed: Optional[Tensor] = None,
                    memory_pos_embed: Optional[Tensor] = None,
                    tgt_shape: Optional[tuple] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, tgt_pos_embed)
        v = tgt2
        h, w = tgt_shape

        if self.tgt_down_scale is not None:
            k = k.view(h, w, k.size(1), -1).permute(2, 3, 0, 1).contiguous()                 # (N, C, H, W)
            v = v.view(h, w, v.size(1), -1).permute(2, 3, 0, 1).contiguous()                 # (N, C, H, W)
            k = F.interpolate(k, scale_factor=1./self.tgt_down_scale, mode='nearest')        # (N, C, H, W)
            v = F.interpolate(v, scale_factor=1./self.tgt_down_scale, mode='nearest')        # (N, C, H, W)
            k = k.view(k.size(0), k.size(1), -1).permute(2, 0, 1).contiguous()               # (H * W, N, C)
            v = v.view(v.size(0), v.size(1), -1).permute(2, 0, 1).contiguous()               # (H * W, N, C)
            if tgt_key_padding_mask is not None:
                tgt_key_padding_mask = tgt_key_padding_mask.view(-1, h, w)[:, None].float()  # (N, 1, H, W)
                tgt_key_padding_mask = F.interpolate(tgt_key_padding_mask, scale_factor=1./self.tgt_down_scale, mode='nearest')  # (N, 1, H, W)
                tgt_key_padding_mask = tgt_key_padding_mask[:, 0].bool().flatten(1)          # (N, H * W)

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_attn_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)

        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos_embed),
        #                            key=self.with_pos_embed(memory, memory_pos_embed),
        #                            value=memory, attn_mask=memory_attn_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]

        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos_embed),
                                         key=self.with_pos_embed(memory, memory_pos_embed),
                                         value=memory, attn_mask=memory_attn_mask,
                                         key_padding_mask=memory_key_padding_mask)
        
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, attn

    def _forward(self,
                 tgt,
                 memory,
                 tgt_attn_mask: Optional[Tensor] = None,
                 memory_attn_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 tgt_pos_embed: Optional[Tensor] = None,
                 memory_pos_embed: Optional[Tensor] = None,
                 tgt_shape: Optional[tuple] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_attn_mask, memory_attn_mask, tgt_key_padding_mask,
                                    memory_key_padding_mask, tgt_pos_embed, memory_pos_embed, tgt_shape)
        return self.forward_post(tgt, memory, tgt_attn_mask, memory_attn_mask, tgt_key_padding_mask,
                                 memory_key_padding_mask, tgt_pos_embed, memory_pos_embed, tgt_shape)

    def forward(self,
                tgt,
                memory,
                tgt_attn_mask: Optional[Tensor] = None,
                memory_attn_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_pos_embed: Optional[Tensor] = None,
                memory_pos_embed: Optional[Tensor] = None,
                tgt_shape: Optional[tuple] = None):
        if self.use_checkpoint and self.training:
            out = checkpoint.checkpoint(self._forward,
                                        tgt,
                                        memory,
                                        tgt_attn_mask,
                                        memory_attn_mask,
                                        tgt_key_padding_mask,
                                        memory_key_padding_mask,
                                        tgt_pos_embed,
                                        memory_pos_embed,
                                        tgt_shape)
        else:
            out = self._forward(tgt,
                                memory,
                                tgt_attn_mask,
                                memory_attn_mask,
                                tgt_key_padding_mask,
                                memory_key_padding_mask,
                                tgt_pos_embed,
                                memory_pos_embed,
                                tgt_shape)
        return out


# import os
# import cv2
# import numpy as np
# for i, attn_img in enumerate(attn):
#     for j, attn_q in enumerate(attn_img):
#         os.makedirs('/data/sets/czy/query_{}'.format(j), exist_ok=True)
#         attn_q = attn_q.reshape(80, 200).cpu().numpy()
#         attn_q = (attn_q - attn_q.min()) / (attn_q.max() - attn_q.min())
#         attn_q = (attn_q * 255).astype(np.uint8)
#         cv2.imwrite(os.path.join('/data/sets/czy/query_{}'.format(j), '{}.png'.format(i)), attn_q)


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
