from torchvision.transforms import Compose
from collections import defaultdict
from .transforms import *


def openlane_transforms_2d(train=True, version='2d', **kwargs):
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    keys = ['img', 'img_mask', 'lane_points', 'lane_attris']

    img_w = 960
    img_h = 480
    cut_height = 320
    thickness = 1
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
    return data_transforms, collate_fn_2d


def openlane_transforms_3d(train=True, version='3d', **kwargs):
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    keys = ['img', 'img_mask', 'bev_mask', 'lane_points', 'lane_attris', 'depth_prior']

    img_w = 960
    img_h = 640
    bev_w = 320
    bev_h = 1600

    bev_range = np.array([-10, 3, 10, 103])
    cut_height = 0
    if train:
        data_transforms = Compose([
            GenerateLaneLine3D(
                img_h, img_w, bev_h, bev_w, bev_range, img_flip_p=0.5, img_rotate=10,
                cut_height=cut_height, theta_x=0, theta_y=0, theta_z=0, tx=0, ty=0, tz=0),
            Normalize(**img_norm_cfg),
            Pad(size_divisor=32),
            ToTensor(keys),
        ])
    else:
        data_transforms = Compose([
            GenerateLaneLine3D(
                img_h, img_w, bev_h, bev_w, bev_range, img_flip_p=0, img_rotate=0,
                cut_height=cut_height, theta_x=0, theta_y=0, theta_z=0, tx=0, ty=0, tz=0),
            Normalize(**img_norm_cfg),
            Pad(size_divisor=32),
            ToTensor(keys),
        ])
    return data_transforms, collate_fn_3d


def collate_fn_2d(data_list):
    result = defaultdict(list)
    for data in data_list:
        img_shape = data['img_metas']['img_shape'][:2]
        if 'img_mask' not in data:
            data['img_mask'] = torch.zeros((0, *img_shape), dtype=torch.uint8)
        elif len(data['img_mask'].shape) == 2:
            data['img_mask'] = data['img_mask'].unsqueeze(0)
        
        if 'lane_attris' not in data:
            data['lane_attris'] = torch.zeros(len(data['lane_points']), dtype=torch.int32)
        
        for key, value in data.items():
            result[key].append(value)
    
    result['img'] = torch.stack(result['img'], dim=0)
    for img_meta in result['img_metas']:
        class_weight = torch.tensor([
            1, 1.10944152, 1., 25.73526669, 10.4877997, 19.31613351, 20.33392954, 4.852459, 3.8657429, 16.94360171, 
            1.87146866, 4.16201426, 4.20553516, 1., 1., 1., 1., 1., 1., 1., 1.05482755], dtype=torch.float32)
        img_meta.update(class_weight=class_weight)
    return result


def collate_fn_3d(data_list):
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
        class_weight = torch.tensor([
            1, 1.10944152, 1., 25.73526669, 10.4877997, 19.31613351, 20.33392954, 4.852459, 3.8657429, 16.94360171, 
            1.87146866, 4.16201426, 4.20553516, 1., 1., 1., 1., 1., 1., 1., 1.05482755], dtype=torch.float32)
        img_meta.update(class_weight=class_weight)
    return result


def openlane_transforms(train=True, version='2d', **kwargs):
    if version == '2d':
        return openlane_transforms_2d(train=train, version=version, **kwargs)
    else:
        return openlane_transforms_3d(train=train, version=version, **kwargs)
