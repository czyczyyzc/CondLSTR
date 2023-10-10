from .fcn import FCN
from .fpn import FPN
from .aspp import ASPP
from .bifpn import BiFPN
from .pspnet import PSPNet
from .upernet import UPerNet
from .trans_fpn import TransConvFPN
from .second_fpn import SECONDFPN
from .imvoxel_neck import OutdoorImVoxelNeck


__factory = {
    'FCN':                FCN,
    'FPN':                FPN,
    'ASPP':               ASPP,
    'BiFPN':              BiFPN,
    'PSPNet':             PSPNet,
    'UPerNet':            UPerNet,
    'SECONDFPN':          SECONDFPN,
    'TransConvFPN':       TransConvFPN,
    'OutdoorImVoxelNeck': OutdoorImVoxelNeck,
}


def names():
    return sorted(__factory.keys())


def create(name=None, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name.
    """
    if name is None:
        name = kwargs.pop('name')
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
