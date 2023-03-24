import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

# class VOCSegmentation(torchvision.datasets.VOCSegmentation):
#     def __init__(self, root, year, image_set, download, transforms=None, target_transform=None):
#         super().__init__(root, year, image_set, download, transforms, target_transform)
#         return

#     def __getitem__(self, index):
#         img, target = super().__getitem__(index)
#         if self.transforms:
#             img = self.transforms(img)
#         if self.target_transform:
#             target = self.target_transform(target)
#         return img, target
# # -*- coding: utf-8 -*-
# """
# Created on Tue Mar 12 23:23:51 2019

# @author: Keshik
# """
# import numpy as np 
# import torch
# import torchvision.datasets.voc as voc
# from torchvision import transforms
# from torch.utils.data import DataLoader


class PascalVOC_Dataset(torchvision.datasets.VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
    """
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        super().__init__(
             root, 
             year=year, 
             image_set=image_set, 
             download=download, 
             transform=transform, 
             target_transform=target_transform)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
    
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        return super().__getitem__(index)
    
    def __len__(self):
        """
        Returns:
            size of the dataset
        """
        return len(self.images)

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

def encode_labels(target):
    """
    Encode multiple labels using 1/0 encoding 
    
    Args:
        target: xml tree file
    Returns:
        torch tensor encoding labels as 1/0 vector
    """
    
    ls = target['annotation']['object']
  
    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))
  
    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))
    
    k = np.zeros(len(object_categories))
    k[j] = 1
  
    return torch.from_numpy(k)

# main 
if __name__ == '__main__':
    voc_dataset = PascalVOC_Dataset(root='~/.data', year='2012', image_set='train', download=False, transform=image_transform, target_transform=encode_labels)
    voc_dataloader = torch.utils.data.DataLoader(voc_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=lambda x: x)

    for i, data in enumerate(voc_dataloader):
        print(data)
        print(data[0][0].shape)
        print(data[0][1].shape)
        break
