from .lane import __factory as __lane_factory

__factory = {
}

__factory.update(__lane_factory)


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name.
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](*args, **kwargs)
