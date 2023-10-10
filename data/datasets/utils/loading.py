import io
import os
import mmcv
import jsonlines
import numpy as np
from PIL import Image, ImageFile
from mmengine.fileio import FileClient
from pycocotools import mask as mask_utils
from ...structures import get_points_type


ImageFile.LOAD_TRUNCATED_IMAGES = True


class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            assert os.path.exists(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


class LoadImageFromFile(object):
    """Load image from a image file.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load image from file.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = mmcv.imread(filename, self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


class LoadImageFromByteString(object):
    """Load image from a byte string.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load image from file.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        byte_str = results['img_byte_str']

        # 创建一个BytesIO对象
        byte_io = io.BytesIO(byte_str)

        # 使用PIL库读取这个字节串为图像
        img = Image.open(byte_io)

        # 将PIL图像对象转换为NumPy数组
        img = np.array(img)
        if len(img.shape) < 3:
            img = img[:, :, None].repeat(3, axis=2)
        img = img[:, :, ::-1]

        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results.pop('img_byte_str')
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


# class LoadImageFromNori(object):
#     """Load image from a image file.
#
#     Args:
#         to_float32 (bool): Whether to convert the img to float32.
#             Defaults to False.
#         color_type (str): Color type of the file. Defaults to 'unchanged'.
#     """
#
#     def __init__(self, to_float32=False, color_type='unchanged'):
#         self.to_float32 = to_float32
#         self.color_type = color_type
#         self.fetcher = nori.Fetcher()
#
#     def __call__(self, results):
#         """Call function to load image from file.
#
#         Args:
#             results (dict): Result dict containing multi-view image filenames.
#
#         Returns:
#             dict: The result dict containing the multi-view image data. \
#                 Added keys and values are described below.
#
#                 - filename (str): Multi-view image filenames.
#                 - img (np.ndarray): Multi-view image arrays.
#                 - img_shape (tuple[int]): Shape of multi-view image arrays.
#                 - ori_shape (tuple[int]): Shape of original image arrays.
#                 - pad_shape (tuple[int]): Shape of padded image arrays.
#                 - scale_factor (float): Scale factor.
#                 - img_norm_cfg (dict): Normalization configuration of images.
#         """
#         img_id = results['sample_idx']
#         filename = results['img_filename']
#         img = cv2.imdecode(np.frombuffer(self.fetcher.get(img_id), np.uint8), cv2.IMREAD_UNCHANGED)
#         # img = img[:, :, ::-1]
#         # img = Image.fromarray(img[:, :, ::-1], mode="RGB")
#         if self.to_float32:
#             img = img.astype(np.float32)
#         results['filename'] = filename
#         # unravel to list, see `DefaultFormatBundle` in formating.py
#         # which will transpose each image separately and then stack into array
#         results['img'] = img
#         results['img_shape'] = img.shape
#         results['ori_shape'] = img.shape
#         # Set initial values for default meta_keys
#         results['pad_shape'] = img.shape
#         results['scale_factor'] = 1.0
#         num_channels = 1 if len(img.shape) < 3 else img.shape[2]
#         results['img_norm_cfg'] = dict(
#             mean=np.zeros(num_channels, dtype=np.float32),
#             std=np.ones(num_channels, dtype=np.float32),
#             to_rgb=False)
#         return results
#
#     def __repr__(self):
#         """str: Return a string that describes the module."""
#         repr_str = self.__class__.__name__
#         repr_str += f'(to_float32={self.to_float32}, '
#         repr_str += f"color_type='{self.color_type}')"
#         return repr_str


# class LoadMultiViewImageFromNori(object):
#     """Load multi channel images from a list of separate channel files.
#
#     Expects results['img_filename'] to be a list of filenames.
#
#     Args:
#         to_float32 (bool): Whether to convert the img to float32.
#             Defaults to False.
#         color_type (str): Color type of the file. Defaults to 'unchanged'.
#     """
#
#     def __init__(self, to_float32=False, color_type='unchanged'):
#         self.to_float32 = to_float32
#         self.color_type = color_type
#         self.fetcher = nori.Fetcher()
#
#     def __call__(self, results):
#         """Call function to load multi-view image from files.
#
#         Args:
#             results (dict): Result dict containing multi-view image filenames.
#
#         Returns:
#             dict: The result dict containing the multi-view image data. \
#                 Added keys and values are described below.
#
#                 - filename (str): Multi-view image filenames.
#                 - img (np.ndarray): Multi-view image arrays.
#                 - img_shape (tuple[int]): Shape of multi-view image arrays.
#                 - ori_shape (tuple[int]): Shape of original image arrays.
#                 - pad_shape (tuple[int]): Shape of padded image arrays.
#                 - scale_factor (float): Scale factor.
#                 - img_norm_cfg (dict): Normalization configuration of images.
#         """
#         img_id = results['sample_idx']
#         filename = results['img_filename']
#         # img is of shape (h, w, c, num_views)
#         img = np.stack(
#             [cv2.imdecode(np.frombuffer(self.fetcher.get(index), np.uint8), cv2.IMREAD_UNCHANGED) for index in img_id], axis=-1)
#         if self.to_float32:
#             img = img.astype(np.float32)
#         results['filename'] = filename
#         # unravel to list, see `DefaultFormatBundle` in formating.py
#         # which will transpose each image separately and then stack into array
#         results['img'] = [img[..., i] for i in range(img.shape[-1])]
#         results['img_shape'] = img.shape
#         results['ori_shape'] = img.shape
#         # Set initial values for default meta_keys
#         results['pad_shape'] = img.shape
#         results['scale_factor'] = 1.0
#         num_channels = 1 if len(img.shape) < 3 else img.shape[2]
#         results['img_norm_cfg'] = dict(
#             mean=np.zeros(num_channels, dtype=np.float32),
#             std=np.ones(num_channels, dtype=np.float32),
#             to_rgb=False)
#         return results
#
#     def __repr__(self):
#         """str: Return a string that describes the module."""
#         repr_str = self.__class__.__name__
#         repr_str += f'(to_float32={self.to_float32}, '
#         repr_str += f"color_type='{self.color_type}')"
#         return repr_str


class LoadMapMaskFromFile(object):

    def __init__(self, mask_key=None):
        self.mask_key = mask_key

    def __call__(self, results):
        map_mask_path = results['map_mask_path']

        if map_mask_path.endswith('.npz'):
            if self.mask_key is not None:
                map_mask_data = np.load(map_mask_path)[self.mask_key]
            else:
                map_mask_data = np.load(map_mask_path)['arr_0']

        elif map_mask_path.endswith('.jsonl'):
            with jsonlines.open(map_mask_path, mode='r') as reader:
                map_mask_list = list(reader)
            map_mask_data = []
            for map_mask in map_mask_list:
                if self.mask_key is not None:
                    map_mask = map_mask[self.mask_key]
                map_mask = mask_utils.decode(map_mask)
                map_mask_data.append(map_mask)
            map_mask_data = np.concatenate(map_mask_data, axis=0)
        else:
            raise NotImplementedError

        results['map_mask'] = map_mask_data
        return results


class LoadImgMaskFromFile(object):

    def __init__(self, mask_key=None):
        self.mask_key = mask_key

    def __call__(self, results):
        img_mask_path = results['img_mask_path']

        if img_mask_path.endswith('.npz'):
            if self.mask_key is not None:
                img_mask_data = np.load(img_mask_path)[self.mask_key]
            else:
                img_mask_data = np.load(img_mask_path)['arr_0']

        elif img_mask_path.endswith('.jsonl'):
            with jsonlines.open(img_mask_path, mode='r') as reader:
                img_mask_list = list(reader)
            img_mask_data = mask_utils.decode([(x[self.mask_key]) for x in img_mask_list])
        else:
            raise NotImplementedError

        results['img_mask'] = img_mask_data
        return results
