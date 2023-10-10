import torch.nn as nn
from .shufflenet import *


class ShuffleNet(nn.Module):

    __factory = {
        0.5: shufflenet_v2_x0_5,
        1.0: shufflenet_v2_x1_0,
        1.5: shufflenet_v2_x1_5,
        2.0: shufflenet_v2_x2_0,
    }

    def __init__(self, ratio, pretrained=True, norm_layer=nn.BatchNorm2d):
        super(ShuffleNet, self).__init__()
        if ratio not in self.__factory:
            raise KeyError("Unsupported ratio:", ratio)
        self.model = self.__factory[ratio](
            pretrained=pretrained,
            norm_layer=norm_layer
        )
        self.out_channels = self.model._stage_out_channels[-1]

    def forward(self, x):
        outputs = []
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)
        x = self.model.conv5(x)
        outputs.append(x)
        return outputs
