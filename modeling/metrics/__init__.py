from .lane import __factory as __lane_factory
from .detection import DetectionMetric
from .semantic_seg import SemanticSegMetric
from .instance_seg import InstanceSegMetric
from .classification import ClassificationMetric


__factory = {
    'detection': DetectionMetric,
    'semantic_seg': SemanticSegMetric,
    'instance_seg': InstanceSegMetric,
    'classification': ClassificationMetric,
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
