import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPModule(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding, dilation, norm_layer=nn.BatchNorm2d):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channels, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)


class Decoder(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 48, 1, bias=False)
        self.bn1 = norm_layer(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)


class ASPP(nn.Module):
    def __init__(self, in_channels=(2048, 256), output_stride=8, norm_layer=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(in_channels[0], 256, 1, padding=0, dilation=dilations[0], norm_layer=norm_layer)
        self.aspp2 = ASPPModule(in_channels[0], 256, 3, padding=dilations[1], dilation=dilations[1], norm_layer=norm_layer)
        self.aspp3 = ASPPModule(in_channels[0], 256, 3, padding=dilations[2], dilation=dilations[2], norm_layer=norm_layer)
        self.aspp4 = ASPPModule(in_channels[0], 256, 3, padding=dilations[3], dilation=dilations[3], norm_layer=norm_layer)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels[0], 256, 1, stride=1, bias=False),
                                             # norm_layer(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = norm_layer(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.decoder = Decoder(in_channels[1])
        self._init_weight()

    def forward(self, inputs):
        x, low_level_feat = inputs[-1], inputs[0]
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.decoder(x, low_level_feat)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)
