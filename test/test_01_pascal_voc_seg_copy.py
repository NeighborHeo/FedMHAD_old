import sys
import os 
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50

config = {
    'voc_dataset_path': "~/.data/",
    'image_size': 224, 
    'batch_size': 32,
    'num_workers': 4,
    'num_classes': 20,
}

# voc segmentation dataset
import numpy as np
import torch

class VOCAnnotationTransform:
    """
    VOC Annotation Transform

    Parameters:
    - class_to_ind (dict): 클래스 이름을 인덱스로 매핑한 딕셔너리
    - keep_difficult (bool): difficult하게 레이블링된 객체를 포함할지 여부
    """
    def __init__(self, class_to_ind=None, keep_difficult=False):
        if class_to_ind is None:
            self.class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        else:
            self.class_to_ind = class_to_ind
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        VOC Annotation Transform을 호출합니다.

        Parameters:
        - target (xml.etree.ElementTree.Element): VOC 데이터셋의 XML 형식으로 된 타겟
        - width (int): 이미지의 폭
        - height (int): 이미지의 높이

        Returns:
        - tuple: 객체 정보, 예) (boxes, labels, is_difficult)
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # rescale
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_idx]
        if len(res) == 0:
            res = np.zeros((1, 5))
        else:
            res = np.array(res)
        return res

class VOCDetectionDataset(Dataset):
    """
    VOC Detection Dataset

    Parameters:
    - root (str): VOC 데이터셋의 루트 디렉토리 경로
    - year (str): 사용할 데이터셋 연도
    - image_set (str): 사용할 데이터셋 세트(train, val, test 중 하나)
    - transform (callable): 데이터셋에 적용할 변환
    - target_transform (callable): 레이블 정보에 적용할 변환
    - keep_difficult (bool): difficult한 객체를 포함할지 여부
    """
    def __init__(self, root, year='2012', image_set='train', transform=None, target_transform=None, keep_difficult=False):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult
        self._voc = VOCDetection(root=self.root, year=self.year, image_set=self.image_set, download=False)
        self._annopath = os.path.join(self.root, 'VOC' + self.year, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(self.root, 'VOC' + self.year, 'JPEGImages', '%s.jpg')
        self.ids = list()
        for line in open(os.path.join(self.root, 'VOC' + self.year, 'ImageSets', 'Main', self.image_set + '.txt')):
            self.ids.append((self.root, line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id[1]).getroot()
        img = Image.open(self._imgpath % img_id[1]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target, img.size[0], img.size[1])
        return img, target

    def __len__(self):
        return len(self.ids)
    
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    transform_train = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])


    # 데이터셋을 224x224로 리사이즈하는 함수 정의
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 데이터셋 다운로드 및 불러오기
    train_dataset = VOCSegmentation(root='~/.data', year='2012', image_set='train', download=False, transforms=transform_train)
    val_dataset = VOCSegmentation(root='~/.data', year='2012', image_set='val', download=False, transforms=transform_train)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    for images, labels in train_loader:
        print(images.shape)
        print(labels.shape)
        break
    
import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
import torch

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = voc_cmap()
    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None):

        is_aug=False
        if year=='2012_aug':
            is_aug = True
            year = '2012'
        
        self.root = os.path.expanduser(root)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.transform = transform
        
        self.image_set = image_set
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        if is_aug and image_set=='train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(mask_dir), "SegmentationClassAug not found, please refer to README.md and prepare it manually"
            split_f = os.path.join( self.root, 'train_aug.txt')#'./datasets/data/train_aug.txt'
        else:
            mask_dir = os.path.join(voc_root, 'SegmentationClass')
            splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)

if __name__ == '__main__':
    dataset = VOCSegmentation(root='~/.data', download=False)
    print(len(dataset))
    loader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for i, (images, masks) in enumerate(loader):
        print(i, images.shape, masks.shape)
        break