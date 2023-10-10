import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from mmcv.cnn import ConvModule
from ..utils import resize


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of fetcher.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class PSPNet(nn.Module):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                 in_channels,
                 channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 align_corners=False,
                 pool_scales=(1, 2, 3, 6),
                 use_checkpoint=False):
        super(PSPNet, self).__init__()

        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales

        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.use_checkpoint = use_checkpoint

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _forward(self, *inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def forward(self, inputs):
        if self.use_checkpoint and self.training:
            output = checkpoint.checkpoint(self._forward, *inputs)
        else:
            output = self._forward(*inputs)
        return output
