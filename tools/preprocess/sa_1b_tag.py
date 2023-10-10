import os
import io
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import math
import copy
import glob
import mmcv
import torch
import pickle
import orjson
import tarfile
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm
from os import path as osp
from PIL import Image, ImageFile
from collections import defaultdict
from pycocotools import mask as mask_utils
from modeling.models.classifiers import open_clip
from modeling.models.classifiers.open_clip.transform import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD, InterpolationMode, ResizeMaxSize, Compose, Normalize
from modeling.models.utils import OPENAI_IMAGENET_TEMPLATES
from modeling.models.classifiers.open_clip import tokenize


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS_TRAIN = {
    'v1.0': {
        'train': 'sa1b-data',
    },
    'mask': {
        'train': 'sa1b-data',
    },
    'tag': {
        'train': 'sa1b-data',
    },
    'mask_tag': {
        'train': 'sa1b-data',
    }
}

DATASETS_VAL = {
    'v1.0': {
        'val': 'sa1b-data',
    },
    'mask': {
        'val': 'sa1b-data',
    },
    'tag': {
        'val': 'sa1b-data',
    },
    'mask_tag': {
        'val': 'sa1b-data',
    }
}


def sa_1b_data_prep(root_path,
                    info_prefix,
                    version='v1.0',
                    num_workers=1,
                    num_gpus=1):
    """Create info file of nuscene dataset.
    Given the raw data, generate its related info file in pkl format.
    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        num_workers: int
    """
    train_data_infos = []
    for dataset, fpath in DATASETS_TRAIN[version].items():
        file_path = os.path.join(root_path, fpath, '*.tar')
        train_files = sorted(glob.glob(file_path))
        train_infos = INfoProcessor(root_path, train_files, dataset, version, num_workers, num_gpus)()
        print('train infos: {}'.format(len(train_infos)))
        train_data_infos.extend(train_infos)
    print('Total train infos in {}: {}'.format(dataset, len(train_data_infos)))
    return


class INfoProcessor(object):

    def __init__(self, root, file_paths, dataset, version='v1.0', num_workers=1, num_gpus=1):
        self.root = root
        self.file_paths = file_paths
        self.dataset = dataset
        self.version = version
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        
        model_path = '/chenziye/project/checkpoints'
        pretrained = os.path.join(model_path, 'open_clip/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin')
        self.clip = open_clip.create_model_and_transforms(
            'convnext_large_d_320', pretrained=pretrained, precision='fp16')[0].eval()
        self.preprocess = Compose([
            Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
            ResizeMaxSize(self.clip.visual.image_size[0], interpolation=InterpolationMode.BICUBIC, fill=0),
        ])

    @staticmethod
    def article(name):
        return "an" if name[0] in "aeiou" else "a"

    @staticmethod
    def processed_name(name, rm_dot=False):
        # _ for lvis
        # / for obj365
        res = name.replace("_", " ").replace("/", " or ").lower()
        if rm_dot:
            res = res.rstrip(".")
        return res

    def _process_single(self, args):
        worker_id, gpu_id = args
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        model = self.clip.to(device)
        file_paths = self.file_paths[worker_id::self.num_workers]
        data_infos = []
        for i, file_path in enumerate(tqdm(file_paths)):
            img_infos = self.obtain_img_infos(file_path, model, device)
            if img_infos is None or len(img_infos) == 0:
                continue
            file_path = file_path.replace(self.root, '').strip('/')
            data_info = {
                'img_infos': img_infos,
                'file_path': file_path
            }
            data_infos.append(data_info)
        return data_infos
    
    @torch.no_grad()
    def obtain_img_infos(self, file_path, model, device):
        img_info_dict = defaultdict(dict)

        # 打开tar文件
        with tarfile.open(file_path) as tar:
            # 遍历tar文件中的每个文件
            for index, member in tqdm(enumerate(tar.getmembers())):
                if not member.isfile():
                    continue
                # 提取文件对象
                f = tar.extractfile(member)
                if f is None:
                    print("Failed to extract file {} from source tarfile {}!".format(member.name, file_path))
                    raise tarfile.ExtractError

                if member.name.endswith('.jpg'):
                    img_id = {
                        'offset': member.offset_data,
                        'size': member.size
                    }
                    img_name = member.name.split('/')[-1]
                    # 读取文件内容
                    content = f.read()
                    # 创建一个BytesIO对象
                    byte_io = io.BytesIO(content)
                    # 使用PIL库读取这个字节串为图像
                    img = Image.open(byte_io)
                    width, height = img.size
                    assert width * height <= 89478485, "image size {} too large for sa1b!".format(img.size)
                    # 将PIL图像对象转换为NumPy数组
                    img = np.array(img)
                    img = torch.from_numpy(img).float().to(device)
                    img_info_dict[os.path.splitext(member.name)[0]].update({
                        'img_id': img_id,
                        'img_name': img_name,
                    })

                if member.name.endswith('.json'):
                    anno_id = {
                        'offset': member.offset_data,
                        'size': member.size
                    }
                    assert member.name.split('/')[-1].replace('.json', '.jpg') == img_name, "image name not match!"

                    # 读取文件内容
                    img_anno = f.read().decode("utf-8")
                    img_anno = orjson.loads(img_anno)['annotations']
                    assert len(img_anno) > 0, "no image mask!"
                    img_area = [x['area'] for x in img_anno]
                    img_idxs = np.argsort(img_area)[-512:]  # 获取N个最大面积的索引
                    img_anno = [img_anno[i] for i in img_idxs]
                    img_bbox = [x['bbox'] for x in img_anno]
                    img_mask = mask_utils.decode([x['segmentation'] for x in img_anno])
                    img_mask = img_mask.transpose((2, 0, 1))
                    img_mask = torch.from_numpy(img_mask).float().to(device)
                    
                    instance_imgs = []
                    for (mask, bbox) in zip(img_mask, img_bbox):
                        mask = mask[:, :, None]
                        x0, y0, w, h = bbox
                        img_h, img_w = img.shape[:2]
                        x1 = min(int(x0 + w * 1.1), img_w)
                        y1 = min(int(y0 + h * 1.1), img_h)
                        x0 = max(int(x0 - w * 0.1), 0)
                        y0 = max(int(y0 - h * 0.1), 0)
                        instance_img = mask * img + (1 - mask) * torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32, device=device) * 255
                        instance_img = instance_img[y0: y1, x0: x1]
                        instance_img = instance_img.permute(2, 0, 1).contiguous().div(255)
                        instance_img = self.preprocess(instance_img).half()
                        instance_imgs.append(instance_img)
                    instance_imgs = torch.stack(instance_imgs, dim=0)
                    instance_imfs = torch.cat([
                        model.encode_image(instance_imgs[i * 128: (i + 1) * 128], normalize=True)
                        for i in range(math.ceil(len(instance_imgs) / 128))], dim=0)
                    
                    tag_path = os.path.join(file_path.replace('-data', '-tags'), img_name.replace('.jpg', '.txt'))
                    with open(tag_path, "r", encoding="utf-8") as f:
                        instance_txts = f.readline().strip().split(',')
                    
                    instance_txfs = []
                    for old_text in instance_txts:
                        txt_pool = []
                        templates = OPENAI_IMAGENET_TEMPLATES
                        use_format = isinstance(templates[0], str)
                        for template in templates:
                            new_text = template.format(old_text) if use_format else template(old_text)
                            txt_pool.append(new_text)
                        txt_pool = tokenize(txt_pool).to(device)                                
                        txt_feat = model.encode_text(txt_pool, normalize=True)                                
                        txt_feat = txt_feat.mean(dim=0)
                        txt_feat = F.normalize(txt_feat, dim=0)                                
                        instance_txfs.append(txt_feat)
                    instance_txfs = torch.stack(instance_txfs, dim=0)
                    instance_prbs = (100.0 * instance_imfs @ instance_txfs.t()).softmax(dim=0)
                    instance_prbs, instance_inds = instance_prbs.max(dim=0)
                    instance_inds = instance_inds.cpu().numpy().tolist()
                    instance_prbs = instance_prbs.cpu().numpy().tolist()
                    
                    instance_tags = ['' for _ in range(len(instance_imgs))]
                    for i, index in enumerate(instance_inds):
                        instance_tags[index] = instance_txts[i]
                    instance_tags = ','.join(instance_tags)
                    
                    sav_path = file_path.replace('-data', '-mask-tags')
                    if not os.path.exists(sav_path):
                        os.makedirs(sav_path)
                    tag_path = os.path.join(sav_path, img_name.replace('.jpg', '.txt'))
                    with open(tag_path, "w", encoding="utf-8") as f:
                        f.write(instance_tags + "\n")
                    
                    img_info_dict[os.path.splitext(member.name)[0]].update({
                        'anno_id': anno_id,
                    })
        return None
    
    def __call__(self):
        if self.num_workers > 1:
            mp.set_start_method('spawn', force=True)
            with mp.Pool(processes=self.num_workers) as pool:
                args_list = [(i, i % self.num_gpus) for i in range(self.num_workers)]
                data_infos = pool.map(self._process_single, args_list)
            data_infos_all = []
            for data_infos_ in data_infos:
                data_infos_all.extend(data_infos_)
        else:
            data_infos_all = self._process_single((0, 0))  # Use GPU 0 for single process
        return data_infos_all



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preprocessing for sa_1b dataset.')
    parser.add_argument('--root', type=str, default='/chenziye/datasets/sa_1b')
    parser.add_argument('--version', type=str, default='mask_tag')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-gpus', type=int, default=8)

    args = parser.parse_args()

    sa_1b_data_prep(
        root_path=args.root,
        info_prefix='sa_1b',
        version=args.version,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
    )


# import cv2
# sav_path = os.path.join('/chenziye/datasets/bboxes', file_path.split('/')[-1])
# os.makedirs(sav_path, exist_ok=True)
# for im, oc, op in zip(instance_imgs, instance_txts, instance_prbs):                                
#     im = im.permute(1, 2, 0) * torch.tensor([58.395, 57.12, 57.375], device=img.device) + \
#         torch.tensor([123.675, 116.28, 103.53], device=img.device)
#     im = im.cpu().numpy().astype(np.uint8)[:, :, ::-1].copy()
#     sav_name = img_name.replace('.jpg', '_{}_{:.2f}.jpg'.format(oc, op))
#     img_path = os.path.join(sav_path, sav_name)
#     cv2.imwrite(img_path, im)

# img = img.cpu().numpy().astype(np.uint8)[:, :, ::-1].copy()
# img_path = os.path.join(sav_path, img_name)
# cv2.imwrite(img_path, img)