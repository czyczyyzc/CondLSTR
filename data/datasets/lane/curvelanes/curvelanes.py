import os
import pickle
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from data.datasets.utils import LoadImageFromFile, LoadImgMaskFromFile
from .preprocess import lane_data_prep


class CurveLanesDataset(Dataset):

    def __init__(self, root, split='train', transform=None, version='v1.0'):
        self.root = root
        self.split = split
        self.transform = transform
        self.version = version
        self.ann_file = os.path.join(self.root, 'curvelanes_infos_{}_{}.pkl'.format(self.split, self.version))
        if not os.path.exists(self.ann_file):
            lane_data_prep(
                root_path=self.root,
                info_prefix='curvelanes',
                version=self.version,
                num_workers=32)
        self.data_infos = self.load_annotations(self.ann_file)
        self.pipeline = Compose([
            LoadImageFromFile(),
            # LoadImgMaskFromFile(),
        ])
        self.test_mode = self.split == 'test'
        self.metadata, self.version = None, None

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        with open(ann_file, 'rb') as file:
            data = pickle.load(file)
        data_infos = data['infos']
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            lane_points=info['lane_points'],
            img_filename=os.path.join(self.root, info['img_path']),
            img_mask_path=os.path.join(self.root, info['img_mask']),
        )
        return input_dict

    # def get_data(self, index):
    #     input_dict = self.get_data_info(index)
    #     input_dict = self.pipeline(input_dict)
    #     if self.transform is not None:
    #         input_dict = self.transform(input_dict)
    #     return input_dict

    # def __getitem__(self, index):
    #     while True:
    #         try:
    #             input_dict = self.get_data(index)
    #             break
    #         except:
    #             print('Get wrong data at index {}!'.format(index))
    #             index = (index + 1) % len(self.data_infos)
    #     return input_dict

    def __getitem__(self, index):
        input_dict = self.get_data_info(index)
        input_dict = self.pipeline(input_dict)
        if self.transform is not None:
            input_dict = self.transform(input_dict)
        return input_dict

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)
