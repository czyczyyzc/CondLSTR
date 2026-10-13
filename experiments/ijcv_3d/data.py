"""OpenLane transforms for the journal 3D configuration."""

from collections import defaultdict

import numpy as np
import torch
from torchvision.transforms import Compose

from data.transforms.lane.lane import GenerateLaneLine3D
from data.transforms.transforms import Normalize, Pad, ToTensor


IMAGE_HEIGHT = 368
IMAGE_WIDTH = 480
BEV_FEATURE_HEIGHT = 50
BEV_FEATURE_WIDTH = 32
OUTPUT_HEIGHT = 400
OUTPUT_WIDTH = 256
BEV_RANGE = np.array([-10, 3, 10, 103], dtype=np.float32)
BEV_Z_SCALE = 20


def validate_geometry(config):
    expected = {
        "image_input_shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "bev_feature_shape": [BEV_FEATURE_HEIGHT, BEV_FEATURE_WIDTH],
        "output_raster_shape": [OUTPUT_HEIGHT, OUTPUT_WIDTH],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key}={config.get(key)!r}; expected {value!r}")


def openlane_transforms_3d(train=True):
    """Build the exact image/BEV preprocessing stated in the revision."""
    normalize = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    )
    keys = ["img", "img_mask", "bev_mask", "lane_points", "lane_attris", "depth_prior"]
    transform = Compose([
        GenerateLaneLine3D(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            OUTPUT_HEIGHT,
            OUTPUT_WIDTH,
            BEV_RANGE,
            BEV_Z_SCALE,
            img_flip_p=0.5 if train else 0.0,
            img_rotate=10 if train else 0.0,
            cut_height=0,
            theta_x=0,
            theta_y=0,
            theta_z=0,
            tx=0,
            ty=0,
            tz=0,
        ),
        Normalize(**normalize),
        Pad(size_divisor=16),
        ToTensor(keys),
    ])
    return transform, collate_fn_3d


def collate_fn_3d(samples):
    result = defaultdict(list)
    for sample in samples:
        image_shape = sample["img_metas"]["img_shape"][:2]
        if "img_mask" not in sample:
            sample["img_mask"] = torch.zeros((0, 2, *image_shape), dtype=torch.uint8)
        else:
            sample["img_mask"] = sample["img_mask"].reshape(-1, 2, *image_shape)

        bev_shape = sample["img_metas"]["bev_shape"][:2]
        if "bev_mask" not in sample:
            sample["bev_mask"] = torch.zeros((0, 2, *bev_shape), dtype=torch.uint8)
        else:
            sample["bev_mask"] = sample["bev_mask"].reshape(-1, 2, *bev_shape)

        if "lane_attris" not in sample:
            sample["lane_attris"] = torch.zeros(len(sample["lane_points"]), dtype=torch.int64)
        for key, value in sample.items():
            result[key].append(value)

    result["img"] = torch.stack(result["img"])
    result["depth_prior"] = torch.stack(result["depth_prior"])
    class_weight = torch.tensor([
        1, 1.10944152, 1, 25.73526669, 10.4877997, 19.31613351, 20.33392954,
        4.852459, 3.8657429, 16.94360171, 1.87146866, 4.16201426, 4.20553516,
        1, 1, 1, 1, 1, 1, 1, 1.05482755,
    ], dtype=torch.float32)
    for metadata in result["img_metas"]:
        metadata["class_weight"] = class_weight
    return result
