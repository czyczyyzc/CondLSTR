import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from data.datasets.utils import LoadImageFromFile


def read_files(root, target_suffix=['.png', '.jpg', '.jpeg'], version='v1.0-test'):
    test_samples = []
    root = '/'.join(root.split('/')[:-1])
    root_prefix = ''
    root_list = root.split('#')
    if len(root_list) > 1:
        root = root_list[0]
        root_prefix = root_list[1]

    def read_file(fpath):
        if any(fpath.endswith(x) for x in target_suffix):
            img_path = fpath
            if os.path.exists(img_path):
                test_samples.append({
                    'img_path': img_path
                })
            else:
                print("File {:s} not found!".format(img_path))
        elif fpath.endswith('.txt'):
            with open(fpath, "r") as f:
                file_list = f.readlines()
            for i, file_path in enumerate(file_list):
                file_path = file_path.strip()
                file_path = os.path.join(root_prefix, file_path)
                if any(file_path.endswith(x) for x in target_suffix):
                    img_path = file_path
                    if os.path.exists(img_path):
                        test_samples.append({
                            'img_path': img_path
                        })
                    else:
                        print("File {:s} not found!".format(img_path))
                else:
                    print("Unsupported file {:s}!".format(file_path))
        else:
            print("Unsupported file {:s}!".format(fpath))
        return

    if os.path.isdir(root):
        walk_generator = os.walk(root)
        for root_path, dirs, files in walk_generator:
            if len(files) < 1:
                continue
            for file in files:
                read_file(os.path.join(root_path, file))
    elif os.path.isfile(root):
        read_file(root)
    else:
        print("Unsupported file {:s}!".format(root))

    print("Find {:d} images in total!".format(len(test_samples)))
    return test_samples


class LaneTestDataset(Dataset):

    def __init__(self, root, split='test', transform=None, version='v1.0-test'):
        self.root = root
        self.split = split
        self.transform = transform
        self.version = version
        self.pipeline = Compose([
            LoadImageFromFile(),
        ])
        self.data_infos = read_files(self.root, target_suffix=['.png', '.jpg', 'jpeg'], version=self.version)
        self.test_mode = self.split == 'test'
        self.version = version
        self.metadata = None

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
            img_filename=os.path.join(self.root, info['img_path']),
        )
        if 'calibration' in info:
            input_dict.update(calibration=info['calibration'])
        return input_dict

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


# def find_files(target_dir, target_suffix=['.png', '.jpg', '.jpeg']):
#     find_res = []
#     walk_generator = os.walk(target_dir)
#     for root_path, dirs, files in walk_generator:
#         if len(files) < 1:
#             continue
#         for file in files:
#             # file_name, suffix_name = os.path.splitext(file)
#             # if any([suffix_name == suffix for suffix in target_suffix]):
#             #     find_res.append(os.path.join(root_path, file))
#             if any(file.endswith(x) for x in target_suffix):
#                 find_res.append(os.path.join(root_path, file))
#     return find_res
#
#
# class LaneBEVTestDataset(Dataset):
#
#     def __init__(self, root, split='test', transform=None, version='v1.0-test', cam_id=15):
#         self.root = root
#         self.split = split
#         self.transform = transform
#         self.version = version
#         self.cam_id = cam_id
#         self.pipeline = Compose([
#             LoadImageFromFile(),
#         ])
#         self.image_list = sorted(find_files(self.root, target_suffix=['.png', '.jpg', 'jpeg']))
#         self.test_mode = self.split == 'test'
#         self.metadata, self.version = None, None
#
#     def get_data_info(self, index):
#         """Get data info according to the given index.
#
#         Args:
#             index (int): Index of the sample data to get.
#
#         Returns:
#             dict: Data information that will be passed to the data \
#                 preprocessing pipelines. It includes the following keys:
#
#                 - sample_idx (str): Sample index.
#                 - pts_filename (str): Filename of point clouds.
#                 - sweeps (list[dict]): Infos of sweeps.
#                 - timestamp (float): Sample timestamp.
#                 - img_filename (str, optional): Image filename.
#                 - lidar2img (list[np.ndarray], optional): Transformations \
#                     from lidar to different cameras.
#                 - ann_info (dict): Annotation info.
#         """
#         img_file = self.image_list[index]
#         # standard protocal modified from SECOND.Pytorch
#         input_dict = dict(
#             sample_idx=index,
#             img_filename=os.path.join(self.root, img_file),
#             cam_id=self.cam_id,
#         )
#         return input_dict
#
#     def __getitem__(self, index):
#         input_dict = self.get_data_info(index)
#         input_dict = self.pipeline(input_dict)
#         if self.transform is not None:
#             input_dict = self.transform(input_dict)
#         return input_dict
#
#     def __len__(self):
#         """Return the length of data infos.
#
#         Returns:
#             int: Length of data infos.
#         """
#         return len(self.image_list)
