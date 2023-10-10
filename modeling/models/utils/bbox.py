import torch
import numpy as np


def get_proj_mat_by_coord_type(img_meta, coord_type):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
            Can be case-insensitive.

    Returns:
        torch.Tensor: transformation matrix.
    """
    coord_type = coord_type.upper()
    mapping = {'LIDAR': 'lidar2img', 'DEPTH': 'depth2img', 'CAMERA': 'cam2img'}
    assert coord_type in mapping.keys()
    return img_meta[mapping[coord_type]]


def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project points from camera coordicates to image coordinates.

    Args:
        points_3d (torch.Tensor): Points in shape (N, 3).
        proj_mat (torch.Tensor): Transformation matrix between coordinates.
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        torch.Tensor: Points in image coordinates with shape [N, 2].
    """
    points_num = list(points_3d.shape)[:-1]

    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    assert len(proj_mat.shape) == 2, 'The dimension of the projection' \
                                     f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
            d1 == 4 and d2 == 4), 'The shape of the projection matrix' \
                                  f' ({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yeilds better results
    points_4 = torch.cat(
        [points_3d, points_3d.new_ones(*points_shape)], dim=-1)
    point_2d = torch.matmul(points_4, proj_mat.t())

    point_2d[:, 2] = torch.clip(point_2d[:, 2], min=1e-5, max=99999)
    point_2d_res = point_2d[:, :2] / point_2d[:, 2:3]

    if with_depth:
        return torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)
    return point_2d_res
