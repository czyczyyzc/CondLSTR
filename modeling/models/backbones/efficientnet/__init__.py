import torch.nn as nn
from .efficientnet import EfficientNet as EffNet


class EfficientNet(nn.Module):

    def __init__(self, compound_coef=0, in_channels=3, out_stride=32, with_head=False,
                 norm_layer=nn.BatchNorm2d, use_checkpoint=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}', in_channels=in_channels,
                                       out_stride=out_stride, with_head=with_head, norm_layer=norm_layer,
                                       use_checkpoint=use_checkpoint)
        if not with_head:
            del model._conv_head
            del model._bn1
            del model._avg_pooling
            del model._dropout
            del model._fc

        self.model = model
        self.out_channels = 1280

    def forward(self, x):
        outputs = []
        endpoints = self.model.extract_endpoints(x)
        for i, (key, value) in enumerate(endpoints.items()):
            # if i > 0:
            #     outputs.append(value)
            outputs.append(value)
        return outputs

