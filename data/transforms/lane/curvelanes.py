from torchvision.transforms import Compose
from .transforms import *
from .utils import collate_fn_padded


def curvelanes_transforms(train=True, version='v1.0', **kwargs):
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    keys = ['img', 'img_mask']

    img_w = 1600  # 1216   # 800
    img_h = 640   # 480    # 320
    cut_height = {
        (1440, 2560, 3): 640,
        (660, 1570, 3): 180,
        (720, 1280, 3): 368,
    }
    thickness = 8
    if train:
        data_transforms = Compose([
            GenerateLaneLine2D(
                transforms=[
                    dict(name='Resize',
                         parameters=dict(size=dict(height=img_h, width=img_w)),
                         p=1.0),
                    dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
                    dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
                    dict(name='MultiplyAndAddToBrightness',
                         parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                         p=0.6),
                    dict(name='AddToHueAndSaturation',
                         parameters=dict(value=(-10, 10)),
                         p=0.7),
                    dict(name='OneOf',
                         transforms=[
                             dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                             dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                         ],
                         p=0.2),
                    dict(name='Affine',
                         parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                                y=(-0.1, 0.1)),
                                         rotate=(-10, 10),
                                         scale=(0.8, 1.2)),
                         p=0.7),
                    dict(name='Resize',
                         parameters=dict(size=dict(height=img_h, width=img_w)),
                         p=1.0),
                ],
                img_h=img_h, img_w=img_w, cut_height=cut_height, thickness=thickness
            ),
            Normalize(**img_norm_cfg),
            Pad(size_divisor=32),
            ToTensor(keys),
        ])
    else:
        data_transforms = Compose([
            GenerateLaneLine2D(
                transforms=[
                    dict(name='Resize',
                         parameters=dict(size=dict(height=img_h, width=img_w)),
                         p=1.0),
                ],
                img_h=img_h, img_w=img_w, cut_height=cut_height, thickness=thickness
            ),
            Normalize(**img_norm_cfg),
            Pad(size_divisor=32),
            ToTensor(keys),
        ])
    return data_transforms, collate_fn_padded
