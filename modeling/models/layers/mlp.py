import torch
import torch.nn as nn
from typing import Type
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule


class MLP(BaseModule):
    """A simple MLP module.

    Pass features (B, C, N) through an MLP.

    Args:
        in_channels (int): Number of channels of input features.
            Default: 18.
        conv_channels (tuple[int]): Out channels of the convolution.
            Default: (256, 256).
        conv_cfg (dict): Config of convolution.
            Default: dict(type='Conv1d').
        norm_cfg (dict): Config of normalization.
            Default: dict(type='BN1d').
        act_cfg (dict): Config of activation.
            Default: dict(type='ReLU').
    """

    def __init__(self,
                 in_channel=18,
                 conv_channels=(256, 256),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.mlp = nn.Sequential()
        prev_channels = in_channel
        for i, conv_channel in enumerate(conv_channels):
            self.mlp.add_module(
                f'layer{i}',
                ConvModule(
                    prev_channels,
                    conv_channels[i],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True))
            prev_channels = conv_channels[i]

    def forward(self, img_features):
        return self.mlp(img_features)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

