from .lane import __factory as __lane_factory


__factory = {
}

__factory.update(__lane_factory)


def names():
    return sorted(__factory.keys())


def create(name, root=None, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)
