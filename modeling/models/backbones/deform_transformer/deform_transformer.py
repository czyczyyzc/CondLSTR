import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from .ops import MSDeformAttn
from .position_encoding import PositionEmbeddingSine
from .position_encoding import PositionEmbeddingLearnedV2 as PositionEmbeddingLearned


class DeformTransformer(nn.Module):
    def __init__(self, in_channels, src_shape=(16, 168), tgt_shape=(32, 32), d_model=256, n_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation='relu', return_intermediate_dec=False, dec_n_points=4, enc_n_points=4,
                 src_pos_encode='sine', tgt_pos_encode='learned', use_checkpoint=False):
        super().__init__()

        if isinstance(in_channels, int):
            in_channels = [in_channels]
        n_levels = len(in_channels)

        if src_shape is not None:
            if isinstance(src_shape[0], int):
                src_shape = [src_shape]
            assert len(src_shape) == len(in_channels)

        self.input_proj = nn.ModuleList()
        for i in range(len(in_channels)):
            self.input_proj.append(nn.Conv2d(in_channels[i], d_model, kernel_size=1))

        encoder_layer = DeformTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation,
                                                      n_levels, n_heads, enc_n_points, use_checkpoint)
        self.encoder = DeformTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation,
                                                      n_levels, n_heads, dec_n_points, use_checkpoint)
        self.decoder = DeformTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.dropout = nn.Dropout(dropout)

        self.t2s_reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

        if src_pos_encode == 'sine':
            self.src_pos_embed = PositionEmbeddingSine(d_model, normalize=True)
            self.src_lvl_embed = nn.Embedding(n_levels, d_model)
        elif src_pos_encode == 'learned':
            self.src_pos_embed = nn.ModuleList(
                [PositionEmbeddingLearned(shape, d_model) for shape in src_shape],
            )
            self.src_lvl_embed = None
        else:
            raise NotImplementedError

        if tgt_pos_encode == 'sine':
            self.tgt_pos_embed = PositionEmbeddingSine(d_model, normalize=True)
        elif tgt_pos_encode == 'learned':
            self.tgt_pos_embed = PositionEmbeddingLearned(tgt_shape, d_model)
        else:
            raise NotImplementedError

        self.src_shape = src_shape
        self.tgt_shape = tgt_shape
        self.src_pos_encode = src_pos_encode
        self.tgt_pos_encode = tgt_pos_encode

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.xavier_uniform_(self.t2s_reference_points.weight, gain=1.0)
        nn.init.constant_(self.t2s_reference_points.bias, 0.)

    @staticmethod
    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def share_encoder(self):
        del self.input_proj
        del self.encoder
        del self.src_pos_embed
        del self.src_lvl_embed

    def forward(self, srcs, src_masks=None):
        encoder_output = self.forward_encoder(srcs, src_masks)
        decoder_output = self.forward_decoder(encoder_output)
        return decoder_output

    def forward_encoder(self, srcs, src_masks=None):
        if not isinstance(srcs, (list, tuple)):
            srcs = [srcs]
        if not isinstance(src_masks, (list, tuple)):
            src_masks = [src_masks]

        if src_masks[0] is None:
            src_masks = []

        src_flatten = []
        src_mask_flatten = []
        src_pos_embed_flatten = []
        src_spatial_shapes = []
        for i, src in enumerate(srcs):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            src_spatial_shapes.append(spatial_shape)

            if len(src_masks) < i + 1:
                src_mask = torch.zeros((bs, h, w), dtype=torch.bool, device=src.device)          # (N, H, W)
                src_masks.append(src_mask)
            else:
                src_mask = src_masks[i]

            if self.src_pos_encode == 'sine':
                src_pos_embed = self.src_pos_embed(src_mask)                                     # (N, C, H, W)
                src_pos_embed = src_pos_embed + self.src_lvl_embed.weight[i].view(-1, 1, 1)      # (N, C, H, W)
            else:
                src_pos_embed = self.src_pos_embed[i](src_mask)                                  # (N, C, H, W)
                
            src = self.input_proj[i](src)                                                        # (N, C, H, W)
            src = src + src_pos_embed                                                            # (N, C, H, W)

            src = src.flatten(2).transpose(1, 2)                                                 # (N, H * W, C)
            src_mask = src_mask.flatten(1)                                                       # (N, H * W)
            src_pos_embed = src_pos_embed.flatten(2).transpose(1, 2)                             # (N, H * W, C)

            src_flatten.append(src)
            src_mask_flatten.append(src_mask)
            src_pos_embed_flatten.append(src_pos_embed)

        src = torch.cat(src_flatten, 1)                                                          # (N, L * H * W, C)
        src_mask = torch.cat(src_mask_flatten, 1)                                                # (N, L * H * W)
        src_pos_embed = torch.cat(src_pos_embed_flatten, 1)                                      # (N, L * H * W, C)
        src_spatial_shapes = torch.as_tensor(src_spatial_shapes, dtype=torch.long,
                                             device=src.device)                                  # (L, 2)
        src_level_start_index = torch.cat((src_spatial_shapes.new_zeros((1, )),
                                           src_spatial_shapes.prod(1).cumsum(0)[:-1]))           # (L,)
        src_valid_ratios = torch.stack([self.get_valid_ratio(m) for m in src_masks], 1)          # (N, L, 2)

        # encoder
        memory = self.encoder(src,
                              src_spatial_shapes,
                              src_level_start_index,
                              src_valid_ratios,
                              src_pos_embed,
                              src_mask)                                                          # (N, H * W, C)
        return memory, src_spatial_shapes, src_level_start_index, src_valid_ratios, src_mask

    def forward_decoder(self, encoder_output):
        memory, src_spatial_shapes, src_level_start_index, src_valid_ratios, src_mask = encoder_output

        tgt_mask = torch.zeros((memory.size(0), *self.tgt_shape),
                               dtype=torch.bool, device=memory.device)                           # (N, H, W)
        tgt_pos_embed = self.tgt_pos_embed(tgt_mask)                                             # (N, C, H, W)
        tgt_pos_embed = tgt_pos_embed.flatten(2).transpose(1, 2)                                 # (N, H * W, C)
        tgt = tgt_pos_embed                                                                      # (N, H * W, C)

        tgt_spatial_shapes = torch.as_tensor(self.tgt_shape, dtype=torch.long,
                                             device=tgt.device).unsqueeze(0)                     # (1, 2)
        tgt_valid_ratios = self.get_valid_ratio(tgt_mask).unsqueeze(1)                           # (N, 1, 2)
        tgt_level_start_index = tgt_spatial_shapes.new_zeros((1,))                               # (1,)
        tgt_mask = tgt_mask.flatten(1)                                                           # (N, 1 * H * W)

        t2s_reference_points = self.t2s_reference_points(tgt).sigmoid()                          # (N, H * W, 2)

        # decoder
        hs = self.decoder(tgt,
                          memory,
                          tgt_pos_embed,
                          t2s_reference_points,
                          tgt_spatial_shapes,
                          src_spatial_shapes,
                          tgt_level_start_index,
                          src_level_start_index,
                          tgt_valid_ratios,
                          src_valid_ratios,
                          tgt_mask,
                          src_mask)                                                              # (M, N, H * W, C)
        hs = hs.transpose(2, 3)                                                                  # (M, N, C, H * W)
        hs = hs.reshape(*hs.shape[:-1], *self.tgt_shape).contiguous()                            # (M, N, C, H, W)
        return tuple(hs)


class DeformTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu',
                 n_levels=4, n_heads=8, n_points=4, use_checkpoint=False):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.use_checkpoint = use_checkpoint

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def _forward(self,
                 src,
                 src_pos_embed,
                 src_reference_points,
                 src_spatial_shapes,
                 src_level_start_index,
                 src_key_padding_mask):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, src_pos_embed), src_reference_points, src,
                              src_spatial_shapes, src_level_start_index, src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src

    def forward(self,
                src,
                src_pos_embed,
                src_reference_points,
                src_spatial_shapes,
                src_level_start_index,
                src_key_padding_mask):
        if self.use_checkpoint and self.training:
            src = checkpoint.checkpoint(self._forward,
                                        src,
                                        src_pos_embed,
                                        src_reference_points,
                                        src_spatial_shapes,
                                        src_level_start_index,
                                        src_key_padding_mask)
        else:
            src = self._forward(src,
                                src_pos_embed,
                                src_reference_points,
                                src_spatial_shapes,
                                src_level_start_index,
                                src_key_padding_mask)
        return src


class DeformTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_valid_ratios,
                src_pos_embed=None,
                src_key_padding_mask=None):

        src_reference_points = self.get_reference_points(src_spatial_shapes, src_valid_ratios, device=src.device)

        output = src
        for _, layer in enumerate(self.layers):
            output = layer(output,
                           src_pos_embed,
                           src_reference_points,
                           src_spatial_shapes,
                           src_level_start_index,
                           src_key_padding_mask)
        return output


class DeformTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu',
                 n_levels=4, n_heads=8, n_points=4, use_checkpoint=False):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        # self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.self_attn = MSDeformAttn(d_model, 1, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.use_checkpoint = use_checkpoint

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward(self,
                 tgt,
                 src,
                 tgt_pos_embed,
                 tgt_reference_points,
                 t2s_reference_points,
                 tgt_spatial_shapes,
                 src_spatial_shapes,
                 tgt_level_start_index,
                 src_level_start_index,
                 tgt_key_padding_mask,
                 src_key_padding_mask):
        # self attention
        # q = k = self.with_pos_embed(tgt, tgt_pos_embed)
        # tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)

        tgt2 = self.self_attn(self.with_pos_embed(tgt, tgt_pos_embed), tgt_reference_points, tgt,
                              tgt_spatial_shapes, tgt_level_start_index, tgt_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_pos_embed), t2s_reference_points, src,
                               src_spatial_shapes, src_level_start_index, src_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt

    def forward(self,
                tgt,
                src,
                tgt_pos_embed,
                tgt_reference_points,
                t2s_reference_points,
                tgt_spatial_shapes,
                src_spatial_shapes,
                tgt_level_start_index,
                src_level_start_index,
                tgt_key_padding_mask,
                src_key_padding_mask):
        if self.use_checkpoint and self.training:
            tgt = checkpoint.checkpoint(self._forward,
                                        tgt,
                                        src,
                                        tgt_pos_embed,
                                        tgt_reference_points,
                                        t2s_reference_points,
                                        tgt_spatial_shapes,
                                        src_spatial_shapes,
                                        tgt_level_start_index,
                                        src_level_start_index,
                                        tgt_key_padding_mask,
                                        src_key_padding_mask)
        else:
            tgt = self._forward(tgt,
                                src,
                                tgt_pos_embed,
                                tgt_reference_points,
                                t2s_reference_points,
                                tgt_spatial_shapes,
                                src_spatial_shapes,
                                tgt_level_start_index,
                                src_level_start_index,
                                tgt_key_padding_mask,
                                src_key_padding_mask)
        return tgt


class DeformTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self,
                tgt,
                src,
                tgt_pos_embed,
                t2s_reference_points,
                tgt_spatial_shapes,
                src_spatial_shapes,
                tgt_level_start_index,
                src_level_start_index,
                tgt_valid_ratios,
                src_valid_ratios,
                tgt_key_padding_mask=None,
                src_key_padding_mask=None):

        tgt_reference_points = self.get_reference_points(tgt_spatial_shapes, tgt_valid_ratios, device=tgt.device)
        t2s_reference_points = t2s_reference_points[:, :, None] * src_valid_ratios[:, None]

        intermediate = []
        output = tgt
        for _, layer in enumerate(self.layers):
            output = layer(output,
                           src,
                           tgt_pos_embed,
                           tgt_reference_points,
                           t2s_reference_points,
                           tgt_spatial_shapes,
                           src_spatial_shapes,
                           tgt_level_start_index,
                           src_level_start_index,
                           tgt_key_padding_mask,
                           src_key_padding_mask)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


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
