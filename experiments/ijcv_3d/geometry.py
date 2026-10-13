"""Map voted BEV pixels/height units back to the evaluator's ego coordinates."""
import numpy as np


def bev_points_to_ego(points, transform):
    points = np.asarray(points)
    homogeneous = np.c_[points, np.ones(len(points))]
    xyz = (np.asarray(transform) @ homogeneous.T).T[:, :3]
    if not np.isfinite(xyz).all():
        raise FloatingPointError('Non-finite decoded lane coordinates')
    return xyz[np.argsort(xyz[:, 1], kind='stable')]
