import sys
import torch
import numpy as np
from torch import Tensor
from typing import Optional, List
from collections import namedtuple
from collections.abc import Sequence


def is_list_of(obj, target_type):
    return isinstance(obj, list) and all(isinstance(element, target_type) for element in obj)


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    # elif isinstance(data, Sequence) and not isinstance(data, str):
    #     return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    elif isinstance(data, (list, tuple)):
        return [to_tensor(x) for x in data]
    elif isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    else:
        return data
        # raise TypeError(f'type {type(data)} cannot be converted to tensor.')


def to_device(data, device='cuda', non_blocking=True):
    device = torch.device(device)
    if isinstance(data, torch.Tensor):
        data = data.to(device=device, non_blocking=non_blocking)
    elif isinstance(data, (list, tuple)):
        data = [to_device(x, device=device, non_blocking=non_blocking) for x in data]
    elif isinstance(data, dict):
        data = {k: to_device(v, device=device, non_blocking=non_blocking) for k, v in data.items()}
    return data


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


# Convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off


# def collate_fn(batch):
#     batch = list(zip(*batch))
#     batch[0] = nested_tensor_from_tensor_list(batch[0])
#     return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def recur_apply(func, lst, depth=0, out_type=list):
    return out_type([recur_apply(func, item, depth-1, out_type)
                     if isinstance(item, (tuple, list)) and depth > 0
                     else func(item) for item in lst])


def namedtuple_with_defaults(name, fields, defaults):
    if sys.version_info.major == 3 and (
            sys.version_info.minor > 7 or
            (sys.version_info.minor == 7 and sys.version_info.micro >= 6)):
        return namedtuple(name, fields, defaults=defaults)
    type_ = namedtuple(name, fields)
    if defaults:
        type_.__new__.__defaults__ = tuple(defaults)
    return type_


class EMA(object):
    def __init__(self, decay=0.999):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.decay) * x + self.decay * self.shadow[name]
        self.shadow[name] = new_average.clone()


