r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""
import re
import torch
import torch.nn.functional as F
import torchvision
import collections
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


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = F.pad(img, (0, padding[-1], 0, padding[-2]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = F.pad(m, (0, padding[-1], 0, padding[-2]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)
    return tensor, mask


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
        tensor[i, ..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return tensor


def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
        return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        if all([x.shape == elem.shape for x in batch]):
            batch = torch.stack(batch, 0, out=out)
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                # raise TypeError(default_collate_err_msg_format.format(elem.dtype))
                return batch
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            return batch
            # raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    else:
        return batch
    raise TypeError(default_collate_err_msg_format.format(elem_type))
