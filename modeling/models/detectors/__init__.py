from .lane import __factory as __lane_factory


__factory = {}

__factory.update(__lane_factory)


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
