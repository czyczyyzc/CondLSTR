from torchvision.transforms import Compose
from .transforms import *
from .utils import collate_fn_padded


def lane_test_transforms(train=False,  version='v1.0-test', **kwargs):
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    keys = ['img']

    img_w = 960  # 800  # 1920
    img_h = 480  # 384  # 1280
    cut_height = 320
    thickness = 1
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
