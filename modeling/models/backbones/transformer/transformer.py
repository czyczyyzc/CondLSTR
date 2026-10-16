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


def _area_downsample_2d(x: Tensor, down_scale) -> Tensor:
    if down_scale is None:
        return x
    if isinstance(down_scale, torch.Tensor):
        down_scale = float(down_scale.item())
    scale = float(down_scale)
    scale_int = int(scale)
    if scale != scale_int or scale_int <= 0:
        raise ValueError(f"Expected a positive integer down_scale for ONNX-friendly area downsample, got {down_scale}.")
    return F.avg_pool2d(x, kernel_size=scale_int, stride=scale_int)


def _resize_key_padding_mask_2d(mask: Tensor, height: int, width: int, down_scale) -> Tensor:
    mask_dtype = mask.dtype
    mask = mask.view(-1, height, width)[:, None].float()
    mask = F.interpolate(mask, scale_factor=1. / down_scale, mode='nearest')[:, 0].flatten(1)
    return mask.bool() if mask_dtype == torch.bool else mask.to(dtype=mask_dtype)


def _resize_key_padding_mask_3d(mask: Tensor, depth: int, height: int, width: int, down_scale) -> Tensor:
    mask_dtype = mask.dtype
    mask = mask.view(-1, depth, height, width).float()
    mask = F.interpolate(mask, scale_factor=1. / down_scale, mode='nearest').flatten(1)
    return mask.bool() if mask_dtype == torch.bool else mask.to(dtype=mask_dtype)


def _as_bool_padding_mask(mask: Tensor) -> Tensor:
    if mask.dtype == torch.bool:
        return mask
    if torch.is_floating_point(mask):
        return mask < 0
    raise TypeError(f"Padding mask should be bool or floating point, but got {mask.dtype}.")


def _prepare_attention_mask(mask: Optional[Tensor], dtype: torch.dtype) -> Optional[Tensor]:
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        if torch.onnx.is_in_onnx_export():
            return mask.to(dtype=dtype) * -10000.0
        return mask
    if not torch.is_floating_point(mask):
        raise TypeError(f"Attention mask should be bool or floating point, but got {mask.dtype}.")
    return mask.to(dtype=dtype)


class Transformer(nn.Module):

    def __init__(
        self,
        in_channels,
        src_shape=None,
        tgt_shape=(32, 32),
        d_model=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        normalize_before=False,
        return_intermediate_dec=False,
        src_pos_encode='sine',
        tgt_pos_encode='learned',
        src_cam_encode=False,
        tgt_cam_encode=False,
        src_con_encode=False,
        tgt_con_encode=False,
        src_add_encode=False,
        tgt_add_encode=False,
        src_down_scale=None,
        tgt_down_scale=None,
        mem_down_scale=None,
        use_fix_encode=False,
        use_input_proj=True,
        use_checkpoint=False):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        if use_input_proj:
            if len(in_channels) >= 2:
                self.src_input_proj = nn.Conv2d(in_channels[0], d_model, kernel_size=1)
                self.tgt_input_proj = nn.Conv2d(in_channels[1], d_model, kernel_size=1)
            else:
                self.src_input_proj = nn.Conv2d(in_channels[0], d_model, kernel_size=1)
                self.tgt_input_proj = nn.Identity()
        else:
            self.src_input_proj = nn.Identity()
            self.tgt_input_proj = nn.Identity()

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

        self.n_heads = n_heads

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

        if src_con_encode:
            self.src_con_embed = PositionEmbeddingLearned(src_shape[-2:], d_model)
        else:
            self.src_con_embed = None

        if tgt_con_encode:
            self.tgt_con_embed = PositionEmbeddingLearned(tgt_shape[-2:], d_model)
        else:
            self.tgt_con_embed = None

        self.src_shape = src_shape
        self.tgt_shape = tgt_shape
        self.src_pos_encode = src_pos_encode
        self.tgt_pos_encode = tgt_pos_encode
        self.src_cam_encode = src_cam_encode
        self.tgt_cam_encode = tgt_cam_encode
        self.src_con_encode = src_con_encode
        self.tgt_con_encode = tgt_con_encode
        self.src_add_encode = src_add_encode
        self.tgt_add_encode = tgt_add_encode
        self.use_fix_encode = use_fix_encode
        self.src_down_scale = src_down_scale
        self.tgt_down_scale = tgt_down_scale
        self.mem_down_scale = mem_down_scale

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def share_encoder(self):
        del self.src_input_proj
        del self.encoder
        del self.src_pos_embed
        del self.src_cam_embed

    def forward(self,
                src=None,
                tgt=None,
                src_mask=None,
                tgt_mask=None,
                src_pos_embed=None,
                tgt_pos_embed=None,
                attn_mask=None):
        encoder_output = self.forward_encoder(src, src_mask, src_pos_embed)
        decoder_output = self.forward_decoder(encoder_output, tgt, tgt_mask, tgt_pos_embed, attn_mask=attn_mask)
        return decoder_output

    def forward_grouped(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_pos_embed=None,
        tgt_pos_embed=None,
    ):
        """Decode independent object groups against one shared image memory."""
        encoder_output = self.forward_encoder(src, src_mask, src_pos_embed)
        return self.forward_grouped_decoder(
            encoder_output,
            tgt=tgt,
            tgt_mask=tgt_mask,
            tgt_pos_embed=tgt_pos_embed,
        )

    def forward_encoder(self, src=None, src_mask=None, src_pos_embed=None):
        if isinstance(src, (list, tuple)):
            src = src[-1]                                                                                      # (N, C, H, W)

        if isinstance(src_mask, (list, tuple)):
            src_mask = src_mask[-1]                                                                            # (N, H, W)

        if src is not None:
            if len(src.shape) == 4:
                bs, _, h_src, w_src = src.shape
                d_src = None
                src = self.src_input_proj(src)                                                                 # (N, C, H, W)
            elif len(src.shape) == 5:
                bs, d_src, _, h_src, w_src = src.shape
                src = self.src_input_proj(src.flatten(0, 1))
                src = src.view(bs, d_src, -1, h_src, w_src)                                                    # (N, D, C, H, W)
            else:
                raise ValueError(f'Invalid src shape: {src.shape}, expected 4D or 5D tensor.')
        elif src_pos_embed is not None:
            if len(src_pos_embed.shape) == 4:
                bs, _, h_src, w_src = src_pos_embed.shape
                d_src = None
            elif len(src_pos_embed.shape) == 5:
                bs, d_src, _, h_src, w_src = src_pos_embed.shape
            else:
                raise ValueError(f'Invalid src_pos_embed shape: {src_pos_embed.shape}, expected 4D or 5D tensor.')
        else:
            h_src, w_src = self.src_shape[-2:]
            d_src = None if len(self.src_shape) == 2 else self.src_shape[0]
        src_shape = (d_src, h_src, w_src)

        if self.use_fix_encode:
            src_mask = None
            if src_pos_embed is None:
                if d_src is None:
                    if self.src_pos_encode == 'sine':
                        src_pos_embed = self.src_pos_embed                                                     # (1, C, H, W)
                        src_pos_embed = src_pos_embed.repeat(bs, 1, 1, 1)                                      # (N, C, H, W)
                    else:
                        src_pos_embed = self.src_pos_embed.pos_embed.weight                                    # (H * W, C)
                        src_pos_embed = src_pos_embed.view(1, h_src, w_src, -1)                                # (1, H, W, C)
                        src_pos_embed = src_pos_embed.permute(0, 3, 1, 2).repeat(bs, 1, 1, 1)                  # (N, C, H, W)
                else:
                    if self.src_pos_encode == 'sine':
                        src_pos_embed = self.src_pos_embed.unsqueeze(1)                                        # (1, 1, C, H, W)
                        src_pos_embed = src_pos_embed.repeat(bs, d_src, 1, 1, 1)                               # (N, D, C, H, W)
                    else:
                        src_pos_embed = self.src_pos_embed.pos_embed.weight                                    # (H * W, C)
                        src_pos_embed = src_pos_embed.view(1, 1, h_src, w_src, -1)                             # (1, 1, H, W, C)
                        src_pos_embed = src_pos_embed.permute(0, 1, 4, 2, 3).repeat(bs, d_src, 1, 1, 1)        # (N, D, C, H, W)

            if self.src_con_encode:
                if d_src is None:
                    src_con_embed = self.src_con_embed.pos_embed.weight                                        # (H * W, C)
                    src_con_embed = src_con_embed.view(1, h_src, w_src, -1)                                    # (1, H, W, C)
                    src_con_embed = src_con_embed.permute(0, 3, 1, 2).repeat(bs, 1, 1, 1)                      # (N, C, H, W)
                    src_pos_embed = src_pos_embed + src_con_embed                                              # (N, C, H, W)
                else:
                    src_con_embed = self.src_con_embed.pos_embed.weight                                        # (H * W, C)
                    src_con_embed = src_con_embed.view(1, 1, h_src, w_src, -1)                                 # (1, 1, H, W, C)
                    src_con_embed = src_con_embed.permute(0, 1, 4, 2, 3).repeat(bs, d_src, 1, 1, 1)            # (N, D, C, H, W)
                    src_pos_embed = src_pos_embed + src_con_embed                                              # (N, D, C, H, W)
        else:
            if src_mask is None:
                if d_src is None:
                    src_mask = torch.zeros((bs, h_src, w_src), dtype=torch.bool, device=src.device)            # (N, H, W)
                else:
                    src_mask = torch.zeros((bs, d_src, h_src, w_src), dtype=torch.bool, device=src.device)     # (N, D, H, W)
            src_pos_mask = _as_bool_padding_mask(src_mask)

            if src_pos_embed is None:
                if d_src is None:
                    src_pos_embed = self.src_pos_embed(src_pos_mask)                                           # (N, C, H, W)
                else:
                    src_pos_embed = self.src_pos_embed(src_pos_mask.flatten(0, 1))                             # (N * D, C, H, W)
                    src_pos_embed = src_pos_embed.view(bs, d_src, -1, h_src, w_src)                            # (N, D, C, H, W)

            if self.src_con_encode:
                if d_src is None:
                    src_con_embed = self.src_con_embed(src_pos_mask)                                           # (N, C, H, W)
                    src_pos_embed = src_pos_embed + src_con_embed                                              # (N, C, H, W)
                else:
                    src_con_embed = self.src_con_embed(src_pos_mask.flatten(0, 1))                             # (N * D, C, H, W)
                    src_con_embed = src_con_embed.view(bs, d_src, -1, h_src, w_src)                            # (N, D, C, H, W)
                    src_pos_embed = src_pos_embed + src_con_embed                                              # (N, D, C, H, W)

        if self.src_cam_encode:
            src_pos_embed = src_pos_embed + self.src_cam_embed.weight[:, :, None, None]                        # (N, D, C, H, W)

        if src is None:
            src = torch.zeros_like(src_pos_embed)                                                              # (N, C, H, W) / (N, D, C, H, W)

        if self.src_add_encode:
            src = src + src_pos_embed                                                                          # (N, C, H, W) / (N, D, C, H, W)

        if d_src is not None:
            src_pos_embed = src_pos_embed.transpose(1, 2)                                                      # (N, C, D, H, W)
            src = src.transpose(1, 2)                                                                          # (N, C, D, H, W)

        src = src.flatten(2).permute(2, 0, 1)                                                                  # (H * W, N, C) / (D * H * W, N, C)
        src_mask = src_mask.flatten(1) if src_mask is not None else src_mask                                   # (N, H * W) / (N, D * H * W)
        src_pos_embed = src_pos_embed.flatten(2).permute(2, 0, 1)                                              # (H * W, N, C) / (D * H * W, N, C)
        src_mask = _prepare_attention_mask(src_mask, src.dtype)

        memory = self.encoder(src,
                              src_key_padding_mask=src_mask,
                              src_pos_embed=src_pos_embed,
                              src_shape=src_shape)                                                             # (H * W, N, C) / (D * H * W, N, C)
        return memory, src_mask, src_pos_embed, src_shape

    def forward_decoder(self, encoder_output, tgt=None, tgt_mask=None, tgt_pos_embed=None, attn_mask=None):
        memory, src_mask, src_pos_embed, src_shape = encoder_output
        bs, c = memory.shape[-2:]
        d_src, h_src, w_src = src_shape

        if tgt is not None:
            if len(tgt.shape) == 4:
                bs, _, h_tgt, w_tgt = tgt.shape
                d_tgt = None
                tgt = self.tgt_input_proj(tgt)                                                                 # (N, C, H, W)
            elif len(tgt.shape) == 5:
                bs, d_tgt, _, h_tgt, w_tgt = tgt.shape
                tgt = self.tgt_input_proj(tgt.flatten(0, 1))
                tgt = tgt.view(bs, d_tgt, -1, h_tgt, w_tgt)                                                    # (N, D, C, H, W)
            else:
                raise ValueError(f'Invalid tgt shape: {tgt.shape}, expected 4D or 5D tensor.')
        elif tgt_pos_embed is not None:
            if len(tgt_pos_embed.shape) == 4:
                bs, _, h_tgt, w_tgt = tgt_pos_embed.shape
                d_tgt = None
            elif len(tgt_pos_embed.shape) == 5:
                bs, d_tgt, _, h_tgt, w_tgt = tgt_pos_embed.shape
            else:
                raise ValueError(f'Invalid tgt_pos_embed shape: {tgt_pos_embed.shape}, expected 4D or 5D tensor.')
        else:
            h_tgt, w_tgt = self.tgt_shape[-2:]
            d_tgt = None if len(self.tgt_shape) == 2 else self.tgt_shape[0]
        tgt_shape = (d_tgt, h_tgt, w_tgt)

        if self.mem_down_scale is not None:
            if d_src is None:
                memory = memory.view(h_src, w_src, bs, -1).permute(2, 3, 0, 1).contiguous()                                        # (N, C, H, W)
                memory = _area_downsample_2d(memory, self.mem_down_scale)                                                           # (N, C, H, W)
                memory = memory.view(bs, c, -1).permute(2, 0, 1).contiguous()                                                      # (H * W, N, C)
                src_pos_embed = src_pos_embed.view(h_src, w_src, bs, -1).permute(2, 3, 0, 1).contiguous()                          # (N, C, H, W)
                src_pos_embed = _area_downsample_2d(src_pos_embed, self.mem_down_scale)                                             # (N, C, H, W)
                src_pos_embed = src_pos_embed.view(bs, c, -1).permute(2, 0, 1).contiguous()                                        # (H * W, N, C)
                if src_mask is not None:
                    src_mask = _resize_key_padding_mask_2d(src_mask, h_src, w_src, self.mem_down_scale)                            # (N, H * W)
            else:
                memory = memory.view(d_src, h_src, w_src, bs, -1).permute(3, 0, 4, 1, 2).contiguous().flatten(0, 1)                # (N * D, C, H, W)
                memory = _area_downsample_2d(memory, self.mem_down_scale)                                                           # (N * D, C, H, W)
                memory = memory.view(bs, d_src, *memory.shape[-3:]).permute(1, 3, 4, 0, 2).contiguous().flatten(0, 2)              # (D * H * W, N, C)
                src_pos_embed = src_pos_embed.view(d_src, h_src, w_src, bs, -1).permute(3, 0, 4, 1, 2).contiguous().flatten(0, 1)  # (N * D, C, H, W)
                src_pos_embed = _area_downsample_2d(src_pos_embed, self.mem_down_scale)                                             # (N * D, C, H, W)
                src_pos_embed = src_pos_embed.view(bs, d_src, *src_pos_embed.shape[-3:]).permute(1, 3, 4, 0, 2).contiguous().flatten(0, 2)  # (D * H * W, N, C)
                if src_mask is not None:
                    src_mask = _resize_key_padding_mask_3d(src_mask, d_src, h_src, w_src, self.mem_down_scale)                     # (N, D * H * W)

        if attn_mask is not None:
            if len(attn_mask.shape) == 4:
                attn_mask = attn_mask.unsqueeze(dim=0).repeat(bs, 1, 1, 1, 1)                                  # (N, H, W, H, W)
            attn_mask = attn_mask.flatten(1, 2).flatten(2, 3)                                                  # (N, H * W, H * W)
            attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)                                       # (N * h, H * W, H * W)

        if self.use_fix_encode:
            tgt_mask = None
            if tgt_pos_embed is None:
                if d_tgt is None:
                    if self.tgt_pos_encode == 'sine':
                        tgt_pos_embed = self.tgt_pos_embed                                                     # (1, C, H, W)
                        tgt_pos_embed = tgt_pos_embed.repeat(bs, 1, 1, 1)                                      # (N, C, H, W)
                    else:
                        tgt_pos_embed = self.tgt_pos_embed.pos_embed.weight                                    # (H * W, C)
                        tgt_pos_embed = tgt_pos_embed.view(1, h_tgt, w_tgt, -1)                                # (1, H, W, C)
                        tgt_pos_embed = tgt_pos_embed.permute(0, 3, 1, 2).repeat(bs, 1, 1, 1)                  # (N, C, H, W)
                else:
                    if self.tgt_pos_encode == 'sine':
                        tgt_pos_embed = self.tgt_pos_embed.unsqueeze(1)                                        # (1, 1, C, H, W)
                        tgt_pos_embed = tgt_pos_embed.repeat(bs, d_tgt, 1, 1, 1)                               # (N, D, C, H, W)
                    else:
                        tgt_pos_embed = self.tgt_pos_embed.pos_embed.weight                                    # (H * W, C)
                        tgt_pos_embed = tgt_pos_embed.view(1, 1, h_tgt, w_tgt, -1)                             # (1, 1, H, W, C)
                        tgt_pos_embed = tgt_pos_embed.permute(0, 1, 4, 2, 3).repeat(bs, d_tgt, 1, 1, 1)        # (N, D, C, H, W)

            if self.tgt_con_encode:
                if d_tgt is None:
                    tgt_con_embed = self.tgt_con_embed.pos_embed.weight                                        # (H * W, C)
                    tgt_con_embed = tgt_con_embed.view(1, h_tgt, w_tgt, -1)                                    # (1, H, W, C)
                    tgt_con_embed = tgt_con_embed.permute(0, 3, 1, 2).repeat(bs, 1, 1, 1)                      # (N, C, H, W)
                    tgt_pos_embed = tgt_pos_embed + tgt_con_embed                                              # (N, C, H, W)
                else:
                    tgt_con_embed = self.tgt_con_embed.pos_embed.weight                                        # (H * W, C)
                    tgt_con_embed = tgt_con_embed.view(1, 1, h_tgt, w_tgt, -1)                                 # (1, 1, H, W, C)
                    tgt_con_embed = tgt_con_embed.permute(0, 1, 4, 2, 3).repeat(bs, d_tgt, 1, 1, 1)            # (N, D, C, H, W)
                    tgt_pos_embed = tgt_pos_embed + tgt_con_embed                                              # (N, D, C, H, W)
        else:
            if tgt_mask is None:
                if d_tgt is None:
                    tgt_mask = torch.zeros((bs, h_tgt, w_tgt), dtype=torch.bool, device=memory.device)         # (N, H, W)
                else:
                    tgt_mask = torch.zeros((bs, d_tgt, h_tgt, w_tgt), dtype=torch.bool, device=memory.device)  # (N, D, H, W)
            if tgt_pos_embed is None:
                tgt_pos_mask = _as_bool_padding_mask(tgt_mask)
                if d_tgt is None:
                    tgt_pos_embed = self.tgt_pos_embed(tgt_pos_mask)                                           # (N, C, H, W)
                else:
                    tgt_pos_embed = self.tgt_pos_embed(tgt_pos_mask.flatten(0, 1))                             # (N * D, C, H, W)
                    tgt_pos_embed = tgt_pos_embed.view(bs, d_tgt, -1, h_tgt, w_tgt)                            # (N, D, C, H, W)

            if self.tgt_con_encode:
                if d_tgt is None:
                    tgt_con_embed = self.tgt_con_embed(tgt_pos_mask)                                           # (N, C, H, W)
                    tgt_pos_embed = tgt_pos_embed + tgt_con_embed                                              # (N, C, H, W)
                else:
                    tgt_con_embed = self.tgt_con_embed(tgt_pos_mask.flatten(0, 1))                             # (N * D, C, H, W)
                    tgt_con_embed = tgt_con_embed.view(bs, d_tgt, -1, h_tgt, w_tgt)                            # (N, D, C, H, W)
                    tgt_pos_embed = tgt_pos_embed + tgt_con_embed                                              # (N, D, C, H, W)

        if self.tgt_cam_encode:
            tgt_pos_embed = tgt_pos_embed + self.tgt_cam_embed.weight[:, :, None, None]                        # (N, D, C, H, W)

        if tgt is None:
            tgt = torch.zeros_like(tgt_pos_embed)                                                              # (N, C, H, W) / (N, D, C, H, W)

        if self.tgt_add_encode:
            tgt = tgt + tgt_pos_embed                                                                          # (N, C, H, W) / (N, D, C, H, W)

        if d_tgt is not None:
            tgt_pos_embed = tgt_pos_embed.transpose(1, 2)                                                      # (N, C, D, H, W)
            tgt = tgt.transpose(1, 2)                                                                          # (N, C, D, H, W)

        tgt = tgt.flatten(2).permute(2, 0, 1)                                                                  # (H * W, N, C) / (D * H * W, N, C)
        tgt_mask = tgt_mask.flatten(1) if tgt_mask is not None else tgt_mask                                   # (N, H * W) / (N, D * H * W)
        tgt_pos_embed = tgt_pos_embed.flatten(2).permute(2, 0, 1)                                              # (H * W, N, C) / (D * H * W, N, C)
        src_mask = _prepare_attention_mask(src_mask, memory.dtype)
        tgt_mask = _prepare_attention_mask(tgt_mask, tgt.dtype)
        attn_mask = _prepare_attention_mask(attn_mask, tgt.dtype)

        hs = self.decoder(tgt,
                          memory,
                          memory_attn_mask=attn_mask,
                          tgt_key_padding_mask=tgt_mask,
                          memory_key_padding_mask=src_mask,
                          tgt_pos_embed=tgt_pos_embed,
                          memory_pos_embed=src_pos_embed,
                          tgt_shape=tgt_shape)                                                                 # (M, H * W, N, C) / (M, D * H * W, N, C)
        if d_tgt is None:
            ys = [x.view(h_tgt, w_tgt, bs, -1).permute(2, 3, 0, 1).contiguous() for x in hs]
        else:
            ys = [x.view(d_tgt, h_tgt, w_tgt, bs, -1).permute(3, 0, 4, 1, 2).contiguous() for x in hs]
        return ys, hs, tgt_mask, tgt_pos_embed

    def forward_grouped_decoder(self, encoder_output, tgt, tgt_mask=None, tgt_pos_embed=None):
        memory, src_mask, src_pos_embed, src_shape = encoder_output
        batch_size, channels = memory.shape[-2:]
        d_src, h_src, w_src = src_shape
        if d_src is not None:
            raise ValueError("Grouped object decoding currently supports 2D image memory only.")
        if tgt.ndim != 5:
            raise ValueError(
                f"Grouped target should have shape (N, M, C, H, W), but got {tuple(tgt.shape)}."
            )

        tgt_batch_size, num_groups, _, h_tgt, w_tgt = tgt.shape
        if tgt_batch_size != batch_size:
            raise ValueError(
                f"Grouped target batch size {tgt_batch_size} does not match image batch size {batch_size}."
            )
        tgt = self.tgt_input_proj(tgt.flatten(0, 1)).view(
            batch_size, num_groups, -1, h_tgt, w_tgt
        )
        if tgt_pos_embed is None:
            raise ValueError("Grouped object decoding requires explicit tgt_pos_embed.")
        if tgt_pos_embed.shape != tgt.shape:
            raise ValueError(
                f"Grouped tgt_pos_embed should match projected target shape {tuple(tgt.shape)}, "
                f"but got {tuple(tgt_pos_embed.shape)}."
            )
        if tgt_mask is None:
            tgt_mask = torch.zeros(
                (batch_size, num_groups, h_tgt, w_tgt),
                dtype=torch.bool,
                device=tgt.device,
            )
        if tgt_mask.shape != (batch_size, num_groups, h_tgt, w_tgt):
            raise ValueError(
                "Grouped tgt_mask should have shape "
                f"{(batch_size, num_groups, h_tgt, w_tgt)}, but got {tuple(tgt_mask.shape)}."
            )

        if self.mem_down_scale is not None:
            memory = memory.view(h_src, w_src, batch_size, -1).permute(2, 3, 0, 1).contiguous()
            memory = _area_downsample_2d(memory, self.mem_down_scale)
            memory = memory.view(batch_size, channels, -1).permute(2, 0, 1).contiguous()
            src_pos_embed = src_pos_embed.view(
                h_src, w_src, batch_size, -1
            ).permute(2, 3, 0, 1).contiguous()
            src_pos_embed = _area_downsample_2d(src_pos_embed, self.mem_down_scale)
            src_pos_embed = src_pos_embed.view(
                batch_size, channels, -1
            ).permute(2, 0, 1).contiguous()
            if src_mask is not None:
                src_mask = _resize_key_padding_mask_2d(
                    src_mask, h_src, w_src, self.mem_down_scale
                )

        tgt_shape = (None, h_tgt, w_tgt)
        tgt = tgt.flatten(3).permute(3, 0, 1, 2).reshape(
            h_tgt * w_tgt, batch_size * num_groups, -1
        )
        tgt_pos_embed = tgt_pos_embed.flatten(3).permute(3, 0, 1, 2).reshape(
            h_tgt * w_tgt, batch_size * num_groups, -1
        )
        tgt_mask = tgt_mask.flatten(2).reshape(
            batch_size * num_groups, h_tgt * w_tgt
        )
        src_mask = _prepare_attention_mask(src_mask, memory.dtype)
        tgt_mask = _prepare_attention_mask(tgt_mask, tgt.dtype)

        hs = self.decoder.forward_grouped(
            tgt,
            memory,
            num_groups=num_groups,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask,
            tgt_pos_embed=tgt_pos_embed,
            memory_pos_embed=src_pos_embed,
            tgt_shape=tgt_shape,
        )
        ys = [
            x.reshape(h_tgt * w_tgt, batch_size, num_groups, -1)
            .permute(1, 2, 3, 0)
            .reshape(batch_size, num_groups, -1, h_tgt, w_tgt)
            .contiguous()
            for x in hs
        ]
        return ys, hs, tgt_mask, tgt_pos_embed


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
        for layer in self.layers:
            output = layer(output,
                           memory,
                           tgt_attn_mask=tgt_attn_mask,
                           memory_attn_mask=memory_attn_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           tgt_pos_embed=tgt_pos_embed,
                           memory_pos_embed=memory_pos_embed,
                           tgt_shape=tgt_shape)
            intermediate.append(self.norm(output))
        return intermediate

    def forward_grouped(
        self,
        tgt,
        memory,
        num_groups,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_pos_embed: Optional[Tensor] = None,
        memory_pos_embed: Optional[Tensor] = None,
        tgt_shape: Optional[tuple] = None,
    ):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer.forward_grouped(
                output,
                memory,
                num_groups=num_groups,
                tgt_attn_mask=tgt_attn_mask,
                memory_attn_mask=memory_attn_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_pos_embed=tgt_pos_embed,
                memory_pos_embed=memory_pos_embed,
                tgt_shape=tgt_shape,
            )
            intermediate.append(self.norm(output))
        return intermediate


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
        d, h, w = src_shape

        if self.src_down_scale is not None:
            if d is None:
                k = k.view(h, w, k.size(1), -1).permute(2, 3, 0, 1).contiguous()                                                     # (N, C, H, W)
                v = v.view(h, w, v.size(1), -1).permute(2, 3, 0, 1).contiguous()                                                     # (N, C, H, W)
                k = _area_downsample_2d(k, self.src_down_scale)                                                                      # (N, C, H, W)
                v = _area_downsample_2d(v, self.src_down_scale)                                                                      # (N, C, H, W)
                k = k.view(k.size(0), k.size(1), -1).permute(2, 0, 1).contiguous()                                                   # (H * W, N, C)
                v = v.view(v.size(0), v.size(1), -1).permute(2, 0, 1).contiguous()                                                   # (H * W, N, C)
                if src_key_padding_mask is not None:
                    src_key_padding_mask = _resize_key_padding_mask_2d(
                        src_key_padding_mask, h, w, self.src_down_scale)
            else:
                k = k.view(d, h, w, k.size(1), -1).permute(3, 0, 4, 1, 2).contiguous().flatten(0, 1)                                 # (N * D, C, H, W)
                v = v.view(d, h, w, v.size(1), -1).permute(3, 0, 4, 1, 2).contiguous().flatten(0, 1)                                 # (N * D, C, H, W)
                k = _area_downsample_2d(k, self.src_down_scale)                                                                      # (N * D, C, H, W)
                v = _area_downsample_2d(v, self.src_down_scale)                                                                      # (N * D, C, H, W)
                k = k.view(-1, d, k.size(1), k.size(2), k.size(3)).permute(1, 3, 4, 0, 2).contiguous().flatten(0, 2)                 # (D * H * W, N, C)
                v = v.view(-1, d, v.size(1), v.size(2), v.size(3)).permute(1, 3, 4, 0, 2).contiguous().flatten(0, 2)                 # (D * H * W, N, C)
                if src_key_padding_mask is not None:
                    src_key_padding_mask = _resize_key_padding_mask_3d(
                        src_key_padding_mask, d, h, w, self.src_down_scale)

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
        d, h, w = src_shape

        if self.src_down_scale is not None:
            if d is None:
                k = k.view(h, w, k.size(1), -1).permute(2, 3, 0, 1).contiguous()                                                     # (N, C, H, W)
                v = v.view(h, w, v.size(1), -1).permute(2, 3, 0, 1).contiguous()                                                     # (N, C, H, W)
                k = _area_downsample_2d(k, self.src_down_scale)                                                                      # (N, C, H, W)
                v = _area_downsample_2d(v, self.src_down_scale)                                                                      # (N, C, H, W)
                k = k.view(k.size(0), k.size(1), -1).permute(2, 0, 1).contiguous()                                                   # (H * W, N, C)
                v = v.view(v.size(0), v.size(1), -1).permute(2, 0, 1).contiguous()                                                   # (H * W, N, C)
                if src_key_padding_mask is not None:
                    src_key_padding_mask = _resize_key_padding_mask_2d(
                        src_key_padding_mask, h, w, self.src_down_scale)
            else:
                k = k.view(d, h, w, k.size(1), -1).permute(3, 0, 4, 1, 2).contiguous().flatten(0, 1)                                 # (N * D, C, H, W)
                v = v.view(d, h, w, v.size(1), -1).permute(3, 0, 4, 1, 2).contiguous().flatten(0, 1)                                 # (N * D, C, H, W)
                k = _area_downsample_2d(k, self.src_down_scale)                                                                      # (N * D, C, H, W)
                v = _area_downsample_2d(v, self.src_down_scale)                                                                      # (N * D, C, H, W)
                k = k.view(-1, d, k.size(1), k.size(2), k.size(3)).permute(1, 3, 4, 0, 2).contiguous().flatten(0, 2)                 # (D * H * W, N, C)
                v = v.view(-1, d, v.size(1), v.size(2), v.size(3)).permute(1, 3, 4, 0, 2).contiguous().flatten(0, 2)                 # (D * H * W, N, C)
                if src_key_padding_mask is not None:
                    src_key_padding_mask = _resize_key_padding_mask_3d(
                        src_key_padding_mask, d, h, w, self.src_down_scale)

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
                                        src_shape,
                                        use_reentrant=False)
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

        if self.tgt_down_scale is not None and self.tgt_down_scale <= 0:
            del self.self_attn
            del self.dropout1
            del self.norm1

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def _merge_grouped_queries(tensor: Tensor, num_groups: int) -> Tensor:
        sequence_length, grouped_batch, channels = tensor.shape
        if grouped_batch % num_groups != 0:
            raise ValueError(
                f"Grouped batch size {grouped_batch} is not divisible by num_groups={num_groups}."
            )
        batch_size = grouped_batch // num_groups
        return tensor.reshape(
            sequence_length, batch_size, num_groups, channels
        ).permute(2, 0, 1, 3).reshape(num_groups * sequence_length, batch_size, channels)

    @staticmethod
    def _split_grouped_queries(
        tensor: Tensor,
        num_groups: int,
        sequence_length: int,
    ) -> Tensor:
        _, batch_size, channels = tensor.shape
        return tensor.reshape(
            num_groups, sequence_length, batch_size, channels
        ).permute(1, 2, 0, 3).reshape(sequence_length, batch_size * num_groups, channels)

    def _grouped_cross_attention(
        self,
        tgt: Tensor,
        memory: Tensor,
        num_groups: int,
        memory_attn_mask: Optional[Tensor],
        memory_key_padding_mask: Optional[Tensor],
        tgt_pos_embed: Optional[Tensor],
        memory_pos_embed: Optional[Tensor],
    ) -> Tensor:
        sequence_length = tgt.shape[0]
        query = self._merge_grouped_queries(
            self.with_pos_embed(tgt, tgt_pos_embed),
            num_groups,
        )
        output = self.multihead_attn(
            query=query,
            key=self.with_pos_embed(memory, memory_pos_embed),
            value=memory,
            attn_mask=memory_attn_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        return self._split_grouped_queries(output, num_groups, sequence_length)

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

        if self.tgt_down_scale is None or self.tgt_down_scale > 0:
            q = k = self.with_pos_embed(tgt, tgt_pos_embed)
            v = tgt
            d, h, w = tgt_shape

            if self.tgt_down_scale is not None:
                if d is None:
                    k = k.view(h, w, k.size(1), -1).permute(2, 3, 0, 1).contiguous()                                                     # (N, C, H, W)
                    v = v.view(h, w, v.size(1), -1).permute(2, 3, 0, 1).contiguous()                                                     # (N, C, H, W)
                    k = _area_downsample_2d(k, self.tgt_down_scale)                                                                      # (N, C, H, W)
                    v = _area_downsample_2d(v, self.tgt_down_scale)                                                                      # (N, C, H, W)
                    k = k.view(k.size(0), k.size(1), -1).permute(2, 0, 1).contiguous()                                                   # (H * W, N, C)
                    v = v.view(v.size(0), v.size(1), -1).permute(2, 0, 1).contiguous()                                                   # (H * W, N, C)
                    if tgt_key_padding_mask is not None:
                        tgt_key_padding_mask = _resize_key_padding_mask_2d(
                            tgt_key_padding_mask, h, w, self.tgt_down_scale)
                else:
                    k = k.view(d, h, w, k.size(1), -1).permute(3, 0, 4, 1, 2).contiguous().flatten(0, 1)                                 # (N * D, C, H, W)
                    v = v.view(d, h, w, v.size(1), -1).permute(3, 0, 4, 1, 2).contiguous().flatten(0, 1)                                 # (N * D, C, H, W)
                    k = _area_downsample_2d(k, self.tgt_down_scale)                                                                      # (N * D, C, H, W)
                    v = _area_downsample_2d(v, self.tgt_down_scale)                                                                      # (N * D, C, H, W)
                    k = k.view(-1, d, k.size(1), k.size(2), k.size(3)).permute(1, 3, 4, 0, 2).contiguous().flatten(0, 2)                 # (D * H * W, N, C)
                    v = v.view(-1, d, v.size(1), v.size(2), v.size(3)).permute(1, 3, 4, 0, 2).contiguous().flatten(0, 2)                 # (D * H * W, N, C)
                    if tgt_key_padding_mask is not None:
                        tgt_key_padding_mask = _resize_key_padding_mask_3d(
                            tgt_key_padding_mask, d, h, w, self.tgt_down_scale)

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_attn_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos_embed),
                                   key=self.with_pos_embed(memory, memory_pos_embed),
                                   value=memory, attn_mask=memory_attn_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_grouped_post(
        self,
        tgt,
        memory,
        num_groups,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_pos_embed: Optional[Tensor] = None,
        memory_pos_embed: Optional[Tensor] = None,
        tgt_shape: Optional[tuple] = None,
    ):
        if self.tgt_down_scale is not None:
            raise ValueError("Grouped object decoding does not support tgt_down_scale.")

        q = k = self.with_pos_embed(tgt, tgt_pos_embed)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_attn_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2 = self._grouped_cross_attention(
            tgt,
            memory,
            num_groups,
            memory_attn_mask,
            memory_key_padding_mask,
            tgt_pos_embed,
            memory_pos_embed,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt

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

        if self.tgt_down_scale is None or self.tgt_down_scale > 0:
            tgt2 = self.norm1(tgt)
            q = k = self.with_pos_embed(tgt2, tgt_pos_embed)
            v = tgt2
            d, h, w = tgt_shape

            if self.tgt_down_scale is not None:
                if d is None:
                    k = k.view(h, w, k.size(1), -1).permute(2, 3, 0, 1).contiguous()                                                     # (N, C, H, W)
                    v = v.view(h, w, v.size(1), -1).permute(2, 3, 0, 1).contiguous()                                                     # (N, C, H, W)
                    k = _area_downsample_2d(k, self.tgt_down_scale)                                                                      # (N, C, H, W)
                    v = _area_downsample_2d(v, self.tgt_down_scale)                                                                      # (N, C, H, W)
                    k = k.view(k.size(0), k.size(1), -1).permute(2, 0, 1).contiguous()                                                   # (H * W, N, C)
                    v = v.view(v.size(0), v.size(1), -1).permute(2, 0, 1).contiguous()                                                   # (H * W, N, C)
                    if tgt_key_padding_mask is not None:
                        tgt_key_padding_mask = _resize_key_padding_mask_2d(
                            tgt_key_padding_mask, h, w, self.tgt_down_scale)
                else:
                    k = k.view(d, h, w, k.size(1), -1).permute(3, 0, 4, 1, 2).contiguous().flatten(0, 1)                                 # (N * D, C, H, W)
                    v = v.view(d, h, w, v.size(1), -1).permute(3, 0, 4, 1, 2).contiguous().flatten(0, 1)                                 # (N * D, C, H, W)
                    k = _area_downsample_2d(k, self.tgt_down_scale)                                                                      # (N * D, C, H, W)
                    v = _area_downsample_2d(v, self.tgt_down_scale)                                                                      # (N * D, C, H, W)
                    k = k.view(-1, d, k.size(1), k.size(2), k.size(3)).permute(1, 3, 4, 0, 2).contiguous().flatten(0, 2)                 # (D * H * W, N, C)
                    v = v.view(-1, d, v.size(1), v.size(2), v.size(3)).permute(1, 3, 4, 0, 2).contiguous().flatten(0, 2)                 # (D * H * W, N, C)
                    if tgt_key_padding_mask is not None:
                        tgt_key_padding_mask = _resize_key_padding_mask_3d(
                            tgt_key_padding_mask, d, h, w, self.tgt_down_scale)

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_attn_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos_embed),
                                   key=self.with_pos_embed(memory, memory_pos_embed),
                                   value=memory, attn_mask=memory_attn_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward_grouped_pre(
        self,
        tgt,
        memory,
        num_groups,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_pos_embed: Optional[Tensor] = None,
        memory_pos_embed: Optional[Tensor] = None,
        tgt_shape: Optional[tuple] = None,
    ):
        if self.tgt_down_scale is not None:
            raise ValueError("Grouped object decoding does not support tgt_down_scale.")

        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, tgt_pos_embed)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_attn_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self._grouped_cross_attention(
            tgt2,
            memory,
            num_groups,
            memory_attn_mask,
            memory_key_padding_mask,
            tgt_pos_embed,
            memory_pos_embed,
        )
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

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

    def _forward_grouped(
        self,
        tgt,
        memory,
        num_groups,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_pos_embed: Optional[Tensor] = None,
        memory_pos_embed: Optional[Tensor] = None,
        tgt_shape: Optional[tuple] = None,
    ):
        if self.normalize_before:
            return self.forward_grouped_pre(
                tgt,
                memory,
                num_groups,
                tgt_attn_mask,
                memory_attn_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                tgt_pos_embed,
                memory_pos_embed,
                tgt_shape,
            )
        return self.forward_grouped_post(
            tgt,
            memory,
            num_groups,
            tgt_attn_mask,
            memory_attn_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            tgt_pos_embed,
            memory_pos_embed,
            tgt_shape,
        )

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
                                        tgt_shape,
                                        use_reentrant=False)
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

    def forward_grouped(
        self,
        tgt,
        memory,
        num_groups,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_pos_embed: Optional[Tensor] = None,
        memory_pos_embed: Optional[Tensor] = None,
        tgt_shape: Optional[tuple] = None,
    ):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(
                self._forward_grouped,
                tgt,
                memory,
                num_groups,
                tgt_attn_mask,
                memory_attn_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                tgt_pos_embed,
                memory_pos_embed,
                tgt_shape,
                use_reentrant=False,
            )
        return self._forward_grouped(
            tgt,
            memory,
            num_groups,
            tgt_attn_mask,
            memory_attn_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            tgt_pos_embed,
            memory_pos_embed,
            tgt_shape,
        )


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


# print('###############################')
# import numpy as np
# import matplotlib.pyplot as plt
# X = tgt[0, :, :, 0].t()
# X = F.normalize(X, dim=1)
# S = (X.mm(X.t()) + 1.0) / 2.0 + 1e-5
# S = S.cpu().numpy()
# M = np.ones_like(S)
# M = np.triu(M, k=1).astype(np.bool)
# S = S[M]
#
# plt.figure(figsize=(6, 4))  ## 设置画布\n",
# plt.hist(S, bins=50)
# plt.xlim(0, 1)
# plt.savefig('/data/gpfs/projects/punim1962/project/count.jpg')
#
# import sys
# sys.exit()


# tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos_embed),
#                                  key=self.with_pos_embed(memory, memory_pos_embed),
#                                  value=memory, attn_mask=memory_attn_mask,
#                                  key_padding_mask=memory_key_padding_mask)
# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# print(attn.shape)
#
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
