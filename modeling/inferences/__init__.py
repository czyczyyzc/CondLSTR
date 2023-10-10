from .lane import __factory as __lane_factory
from .detection import DetectionInference
from .instance_seg import InstanceSegInference
from .classification import ClassificationInference


__factory = {
    'detection': DetectionInference,
    'instance_seg': InstanceSegInference,
    'classification': ClassificationInference,
}

__factory.update(__lane_factory)


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
    """
    if name not in __factory:
        raise KeyError("Unknown task:", name)
    return __factory[name](*args, **kwargs)
