import torch
import torch.nn as nn
from .model_stages import BiSeNet


class STDCNet(nn.Module):

    model_urls = {
        'STDCNet1446': 's3://czyczyyzc/models/stdcnet/STDCNet1446_76.47.tar',
        'STDCNet813':  's3://czyczyyzc/models/stdcnet/STDCNet813M_73.91.tar',
    }

    def __init__(self, backbone, pretrained=True, norm_layer=nn.BatchNorm2d, use_checkpoint=False):
        super(STDCNet, self).__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint
        self.model = BiSeNet(backbone, n_classes=32, norm_layer=norm_layer, use_checkpoint=use_checkpoint)
        self.init_weights()

    def init_weights(self):
        if self.pretrained:
            model_path = self.model_urls[self.backbone]
            with open(model_path, 'rb') as f:
                state_dict = torch.load(f, map_location='cpu')["state_dict"]
            ret = self.model.cp.backbone.load_state_dict(state_dict, strict=False)
            print('Backbone missing_keys: {}'.format(ret.missing_keys))
            print('Backbone unexpected_keys: {}'.format(ret.unexpected_keys))
        return

    def get_param_groups(self, lr, weight_decay=None):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = self.model.get_params()
        param_groups = [
            {'params': wd_params, 'lr': lr * 0.1, 'weight_decay': weight_decay},
            {'params': nowd_params, 'lr': lr * 0.1, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': lr_mul_nowd_params, 'lr': lr, 'weight_decay': 0},
        ]
        return param_groups

    def forward(self, x):
        out = self.model(x)
        return out
