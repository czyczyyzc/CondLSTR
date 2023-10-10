import re
import torch
from torch import Tensor
from typing import Optional, List
from collections import defaultdict


np_str_obj_array_pattern = re.compile(r'[SaUO]')


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # if torchvision._is_tracing():
    #     # nested_tensor_from_tensor_list() does not export well to ONNX
    #     # call _onnx_nested_tensor_from_tensor_list() instead
    #     return _onnx_nested_tensor_from_tensor_list(tensor_list)
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    for i, img in enumerate(tensor_list):
        if len(img.shape) == 1:
            tensor[i, ..., :img.shape[-1]].copy_(img)
        else:
            tensor[i, ..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return tensor


def collate_fn(data_list):
    result = defaultdict(list)
    for data in data_list:
        for key, value in data.items():
            result[key].append(value)
    return result


def collate_fn_padded(data_list, keys=['img', 'img_mask']):
    result = defaultdict(list)
    for data in data_list:
        for key, value in data.items():
            result[key].append(value)
    for key in result:
        if key not in keys:
            continue
        if not all([isinstance(x, torch.Tensor) for x in result[key]]):
            continue
        # result[key] = torch.stack(result[key], dim=0)
        result[key] = nested_tensor_from_tensor_list(result[key])
    return result

