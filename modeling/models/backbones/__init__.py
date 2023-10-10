from .dla import DLAWrapper
from .hrnet import HRNet
from .second import SECOND
from .resnet import ResNet
from .regnet import RegNet
from .resnext import ResNeXt
from .stdcnet import STDCNet
from .shufflenet import ShuffleNet
from .efficientnet import EfficientNet
from .transformer import Transformer
from .swap_transformer import SwapTransformer
from .swin_transformer import SwinTransformer
# from .deform_transformer import DeformTransformer
# from .deform_transformer_v2 import DeformTransformerV2


__factory = {
    'HRNet':               HRNet,
    'ResNet':              ResNet,
    'RegNet':              RegNet,
    'ResNeXt':             ResNeXt,
    'STDCNet':             STDCNet,
    'DLAWrapper':          DLAWrapper,
    'ShuffleNet':          ShuffleNet,
    'SECOND':              SECOND,
    'EfficientNet':        EfficientNet,
    'Transformer':         Transformer,
    'SwapTransformer':     SwapTransformer,
    'SwinTransformer':     SwinTransformer,
    # 'MLPMixer':            MLPMixer,
    # 'DeformTransformer':   DeformTransformer,
    # 'DeformTransformerV2': DeformTransformerV2
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

