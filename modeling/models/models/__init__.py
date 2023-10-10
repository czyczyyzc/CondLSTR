from .lane import __factory as __lane_factory


__factory = {
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
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
