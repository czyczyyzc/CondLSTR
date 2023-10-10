from .focal_loss import FocalLoss, FocalLossV2
from .dice_loss import BinaryDiceLoss
from .binary_seg_kd import BinarySegLossKD


__factory = {
    'FocalLoss':       FocalLoss,
    'BinaryDiceLoss':  BinaryDiceLoss,
    'BinarySegLossKD': BinarySegLossKD
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
