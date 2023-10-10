from torchvision.transforms import Compose
from .transforms import *
from .utils import collate_fn_padded


def once_3dlanes_transforms(train=True, version='v1.0', **kwargs):
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    keys = ['img', 'img_mask', 'map_mask']

    img_w = 480  # 960
    img_h = 384  # 640
    bev_z = 128
    bev_w = 256  # 512
    bev_h = 512  # 1024

    bev_range = np.array([-10, 3, -5, 10, 103, 5])
    cut_height = 0  # 320
    thickness = 1
    if train:
        data_transforms = Compose([
            GenerateLaneLine3D(
                img_h, img_w, bev_h, bev_w, bev_z, bev_range, img_flip_p=0.5, img_rotate=10,
                cut_height=cut_height, theta_x=10, theta_y=10, theta_z=10,
                tx=0, ty=0, tz=0, thickness=thickness
            ),
            Normalize(**img_norm_cfg),
            Pad(size_divisor=32),
            ToTensor(keys),
        ])
    else:
        data_transforms = Compose([
            GenerateLaneLine3D(
                img_h, img_w, bev_h, bev_w, bev_z, bev_range, img_flip_p=0, img_rotate=0,
                cut_height=cut_height, theta_x=0, theta_y=0, theta_z=0,
                tx=0, ty=0, tz=0, thickness=thickness
            ),
            Normalize(**img_norm_cfg),
            Pad(size_divisor=32),
            ToTensor(keys),
        ])
    return data_transforms, collate_fn_padded
