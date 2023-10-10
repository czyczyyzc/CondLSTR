import math
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.nn.init import trunc_normal_
from ...layers import DropPath, Mlp
from .utils import round_width


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = (
        tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    )

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  #  tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
        # If True, perform pool before projection.
        pool_first=False,
    ):
        super().__init__()
        self.pool_first = pool_first
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if np.prod(kernel_q) == 1 and np.prod(stride_q) == 1:
            kernel_q = ()
        if np.prod(kernel_kv) == 1 and np.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool3d if mode == "max" else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv":
            self.pool_q = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(head_dim) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(head_dim) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(head_dim) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

    def forward(self, x, thw_shape):
        B, N, C = x.shape
        if self.pool_first:
            x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(
                0, 2, 1, 3
            )
            q = k = v = x
        else:
            q = k = v = x
            q = (
                self.q(q)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            k = (
                self.k(k)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            v = (
                self.v(v)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        if self.pool_first:
            q_N = (
                np.prod(q_shape) + 1
                if self.has_cls_embed
                else np.prod(q_shape)
            )
            k_N = (
                np.prod(k_shape) + 1
                if self.has_cls_embed
                else np.prod(k_shape)
            )
            v_N = (
                np.prod(v_shape) + 1
                if self.has_cls_embed
                else np.prod(v_shape)
            )

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
            q = (
                self.q(q)
                .reshape(B, q_N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
            v = (
                self.v(v)
                .reshape(B, v_N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
            k = (
                self.k(k)
                .reshape(B, k_N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        N = q.shape[2]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
        pool_first=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool3d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
            if len(kernel_skip) > 0
            else None
        )

    def forward(self, x, thw_shape):
        x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed
        )
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new


class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=(1, 16, 16),
        stride=(1, 4, 4),
        padding=(1, 7, 7),
        conv_2d=False,
    ):
        super().__init__()
        if conv_2d:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        self.proj = conv(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.proj(x)
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2)


class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)
        return x


# PATCH_2D: True
# ZERO_DECAY_POS_CLS: False
# MODE: "conv"
# CLS_EMBED_ON: True
# PATCH_KERNEL: [7, 7]
# PATCH_STRIDE: [4, 4]
# PATCH_PADDING: [3, 3]
# EMBED_DIM: 96
# NUM_HEADS: 1
# MLP_RATIO: 4.0
# QKV_BIAS: True
# DROPPATH_RATE: 0.1
# DEPTH: 16
# NORM: "layernorm"
# DIM_MUL: [[1, 2.0], [3, 2.0], [14, 2.0]]
# HEAD_MUL: [[1, 2.0], [3, 2.0], [14, 2.0]]
# POOL_KVQ_KERNEL: [1, 3, 3]
# POOL_KV_STRIDE_ADAPTIVE: [1, 4, 4]
# POOL_Q_STRIDE: [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]


class MViT(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    Args:
        in_channels:
        depth:              Depth of the transformer.
        mode:               Options include `conv`, `max`.
        norm:               Normalization layer for the transformer. Only layernorm is supported now.
        spatial_size:
        temporal_size:
        pool_first:         If True, perform pool before projection in attention.
        patch_2d:           If True, use 2d patch, otherwise use 3d patch.
        patch_kernel:       Kernel size for patchtification.
        patch_stride:       Stride size for patchtification.
        patch_padding:      Padding size for patchtification.
        pool_q_stride:      Stride size for the Pool Q at layer i.
                            Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
        pool_kv_stride:     Stride size for the Pool KV at layer i.
                            Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
        pool_kv_stride_adaptive: Initial stride size for KV at layer 1. The stride size will be further reduced with
                            the raio of MVIT.DIM_MUL. If will overwrite MVIT.POOL_KV_STRIDE if not None.
        pool_kvq_kernel:    If not None, overwrite the KV_KERNEL and Q_KERNEL size with POOL_KVQ_CONV_SIZ.
                            Otherwise the kernel_size is [s + 1 if s > 1 else s for s in stride_size].
        dim_mul_list:       Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
                            the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
        head_mul_list:      Head number multiplication at layer i. If 2.0 is used, then the next block will
                            increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
        embed_dim:          Base embedding dimension for the transformer.
        num_heads:          Base num of heads for the transformer.
        mlp_ratio:          Dimension reduction ratio for the MLP layers.
        qkv_bias:           If use, use bias term in attention fc layers.
        norm_stem:          If True, use norm after stem.
        dropout_rate:       Dropout rate for the MViT backbone.
        drop_path_rate:     Drop path rate for the tranfomer.
        cls_embed_on:       If True, use cls embed in the network, otherwise don't.
        sep_pos_embed:      If True, perform separate positional embedding.
        zero_decay_pos_cls: If True, perform no decay on positional embedding and cls embedding.
        head_act:           Activation layer for the output head.
        num_classes:
    """

    def __init__(self,
                 in_channels=3,
                 depth=16,
                 mode='conv',
                 norm='layernorm',
                 spatial_size=224,
                 temporal_size=1,
                 pool_first=False,
                 patch_2d=False,
                 patch_kernel=[3, 7, 7],
                 patch_stride=[2, 4, 4],
                 patch_padding=[2, 4, 4],
                 pool_q_stride=[],
                 pool_kv_stride=None,
                 pool_kv_stride_adaptive=None,
                 pool_kvq_kernel=None,
                 dim_mul_list=[],
                 head_mul_list=[],
                 embed_dim=96,
                 num_heads=1,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 norm_stem=False,
                 dropout_rate=0.0,
                 drop_path_rate=0.1,
                 cls_embed_on=True,
                 sep_pos_embed=False,
                 zero_decay_pos_cls=True,
                 head_act='softmax',
                 num_classes=1000,
                 use_checkpoint=False,
                 ):
        super().__init__()
        # Get parameters.
        # Prepare input.
        use_2d_patch = patch_2d
        self.patch_stride = patch_stride
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        # Prepare output.
        # Prepare backbone
        self.drop_rate = dropout_rate
        self.cls_embed_on = cls_embed_on
        self.sep_pos_embed = sep_pos_embed
        self.zero_decay_pos_cls = zero_decay_pos_cls
        if norm == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(
            dim_in=in_channels,
            dim_out=embed_dim,
            kernel=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
            conv_2d=use_2d_patch,
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patch_dims[0], embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, embed_dim)
                )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_dim, embed_dim)
            )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(dim_mul_list)):
            dim_mul[dim_mul_list[i][0]] = dim_mul_list[i][1]
        for i in range(len(head_mul_list)):
            head_mul[head_mul_list[i][0]] = head_mul_list[i][1]

        pool_q = [[] for i in range(depth)]
        pool_kv = [[] for i in range(depth)]
        stride_q = [[] for i in range(depth)]
        stride_kv = [[] for i in range(depth)]

        for i in range(len(pool_q_stride)):
            stride_q[pool_q_stride[i][0]] = pool_q_stride[i][1:]
            if pool_kvq_kernel is not None:
                pool_q[pool_q_stride[i][0]] = pool_kvq_kernel
            else:
                pool_q[pool_q_stride[i][0]] = [
                    s + 1 if s > 1 else s for s in pool_q_stride[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if pool_kv_stride_adaptive is not None:
            _stride_kv = pool_kv_stride_adaptive
            pool_kv_stride = []
            for i in range(depth):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                pool_kv_stride.append([i] + _stride_kv)

        for i in range(len(pool_kv_stride)):
            stride_kv[pool_kv_stride[i][0]] = pool_kv_stride[i][1:]
            if pool_kvq_kernel is not None:
                pool_kv[pool_kv_stride[i][0]] = pool_kvq_kernel
            else:
                pool_kv[pool_kv_stride[i][0]] = [s + 1 if s > 1 else s for s in pool_kv_stride[i][1:]]

        self.norm_stem = norm_layer(embed_dim) if norm_stem else None

        self.blocks = nn.ModuleList()

        self.use_checkpoint = use_checkpoint

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(
                embed_dim,
                dim_mul[i + 1],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
            )
            self.blocks.append(attention_block)

        embed_dim = dim_out
        self.norm = norm_layer(embed_dim)

        self.head = TransformerBasicHead(
            embed_dim,
            num_classes,
            dropout_rate=dropout_rate,
            act_func=head_act,
        )
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            if self.cls_embed_on:
                trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.zero_decay_pos_cls:
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                        "cls_token",
                    }
                else:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                    }
            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}

    def forward(self, x):
        x = x[0]
        x = self.patch_embed(x)

        T = self.DATA.NUM_FRAMES // self.patch_stride[0]
        H = self.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        W = self.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x, thw = checkpoint.checkpoint(blk, x, thw)
            else:
                x, thw = blk(x, thw)

        x = self.norm(x)
        if self.cls_embed_on:
            x = x[:, 0]
        else:
            x = x.mean(1)

        x = self.head(x)
        return x
