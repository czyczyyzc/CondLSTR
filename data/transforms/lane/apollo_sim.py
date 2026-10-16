from torchvision.transforms import Compose
from collections import defaultdict
from .lane import *


def apollo_sim_transforms(train=True, version='v1.0', **kwargs):
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    keys = ['img', 'img_mask', 'bev_mask', 'lane_points', 'lane_attris', 'depth_prior']

    img_w = 480
    img_h = 368
    bev_w = 256
    bev_h = 400
    bev_range = np.array([-10, 3, 10, 103])
    bev_z_scale = 20
    cut_height = 0
    if train:
        data_transforms = Compose([
            GenerateLaneLine3D(
                img_h, img_w, bev_h, bev_w, bev_range, bev_z_scale, img_flip_p=0.5, img_rotate=10,
                cut_height=cut_height, theta_x=0, theta_y=0, theta_z=0, tx=0, ty=0, tz=0),
            Normalize(**img_norm_cfg),
            Pad(size=(img_h, img_w)),
            ToTensor(keys),
        ])
    else:
        data_transforms = Compose([
            GenerateLaneLine3D(
                img_h, img_w, bev_h, bev_w, bev_range, bev_z_scale, img_flip_p=0, img_rotate=0,
                cut_height=cut_height, theta_x=0, theta_y=0, theta_z=0, tx=0, ty=0, tz=0),
            Normalize(**img_norm_cfg),
            Pad(size=(img_h, img_w)),
            ToTensor(keys),
        ])
    return data_transforms, collate_fn


def collate_fn(data_list):
    result = defaultdict(list)
    for data in data_list:
        img_shape = data['img_metas']['img_shape'][:2]
        if 'img_mask' not in data:
            data['img_mask'] = torch.zeros((0, 2, *img_shape), dtype=torch.uint8)
        else:
            data['img_mask'] = data['img_mask'].reshape(-1, 2, *img_shape)

        bev_shape = data['img_metas']['bev_shape'][:2]
        if 'bev_mask' not in data:
            data['bev_mask'] = torch.zeros((0, 2, *bev_shape), dtype=torch.uint8)
        else:
            data['bev_mask'] = data['bev_mask'].reshape(-1, 2, *bev_shape)

        if 'lane_attris' not in data:
            data['lane_attris'] = torch.zeros(len(data['lane_points']), dtype=torch.int32)

        for key, value in data.items():
            result[key].append(value)

    result['img'] = torch.stack(result['img'], dim=0)
    result['depth_prior'] = torch.stack(result['depth_prior'], dim=0)
    for img_meta in result['img_metas']:
        class_weight = torch.tensor([1], dtype=torch.float32)
        img_meta.update(class_weight=class_weight)
    return result
