import pickle
import os 
import numpy as np 
import pandas as pd
import random
from PIL import Image
import cv2
import logging

import torch
from torch.utils.data import Dataset
from torchvision import transforms,datasets
from sklearn.model_selection import StratifiedShuffleSplit
import sys 
sys.path.append("..")
import utils.utils as utils
from torch.utils.data import DataLoader

class myImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index
    
class mydataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, train=False, verbose=False, transforms=None):
        self.img = imgs
        self.gt = labels
        self.train = train
        self.verbose = verbose
        self.aug = False
        self.transforms = transforms
        return
    def __len__(self):
        return len(self.img)
    def __getitem__(self, idx):
        img = self.img[idx]
        gt = self.gt[idx]
        # print(img.shape) # 3, 224, 224
        if self.transforms:
            img = self.transforms(img)
        idx = torch.tensor(0)
        return img, gt, idx
    def get_labels(self):
        return self.gt


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
transformations_train = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomChoice([
                                    transforms.ColorJitter(brightness=(0.80, 1.20)),
                                    transforms.RandomGrayscale(p = 0.25)
                                    ]),
                                transforms.RandomHorizontalFlip(p = 0.25),
                                transforms.RandomRotation(25),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = mean, std = std),
                            ])
transformations_valid = transforms.Compose([transforms.ToPILImage(),
                                        transforms.CenterCrop(224), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean = mean, std = std),
                                        ])

transformations_test = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(255), 
                                        transforms.FiveCrop(224), 
                                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean = mean, std = std)(crop) for crop in crops])),
                                        ])

def dirichlet_datasplit(args, privtype='cifar10', publictype='cifar100', N_parties=20, online=True, public_percent=1):
    #public cifar100
    print(privtype, publictype)
    if publictype== 'cifar100':
        public_dataset = Cifar_Dataset( 
            os.path.join(args.datapath, 'cifar-100-python/'), publictype, train=True, verbose=False, distill = True, aug=online, public_percent=public_percent)
        # public_data = {}
        # public_data['x'] = public_dataset.img
        # import ipdb; ipdb.set_trace()
        # public_data['y'] = public_dataset.gt
        # public_data = public_dataset.img
    elif publictype== 'imagenet': #public_percent not valid
        public_dataset = myImageFolder(
            os.path.join(args.datapath, 'imagenet/train/'),
            transforms.Compose([
                transforms.RandomResizedCrop(32), #224
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        )
    elif publictype== 'mscoco':
        import pathlib
        path = pathlib.Path(args.datapath).joinpath('PASCAL_VOC_2012')
        public_imgs = np.load(path.joinpath('PASCAL_VOC_val_224_Img.npy'))
        public_labels = np.load(path.joinpath('PASCAL_VOC_val_224_Label.npy'))
        # print("size of public dataset: ", public_imgs.shape, "images")
        public_dataset = mydataset(public_imgs, public_labels) # transforms=transformations_valid)
        test_dataset = mydataset(public_imgs.copy(), public_labels.copy()) # transforms=transformations_valid)
        

    distill_loader = DataLoader(
            dataset=public_dataset, batch_size=args.disbatchsize, shuffle=online, 
            num_workers=args.num_workers, pin_memory=True, sampler=None)
    #private
    if privtype=='cifar10':
        subpath = 'cifar-10-batches-py/'
        N_class = 10
    elif privtype=='cifar100':
        subpath = 'cifar-100-python/'
        N_class = 100
    elif privtype=='pascal_voc2012':
        subpath = f'PASCAL_VOC_2012/dirichlet/alpha_{args.alpha:.0f}/'
        N_class = 20
    
    splitname = f'./splitfile/{privtype}/{args.alpha}_{args.seed}.npy'
    if os.path.exists(splitname):
        split_arr =  np.load(splitname)
        # assert split_arr.shape == (N_class, N_parties)
        print(f'size of split_arr: {split_arr.shape}')
    else:
        split_arr = np.random.dirichlet([args.alpha]*N_parties, N_class)#nclass*N_parties
        pathlib.Path(f'./splitfile/{privtype}').mkdir(parents=True, exist_ok=True)
        np.save(splitname, split_arr)
    
    if privtype=="cifar10" or privtype=="cifar100":
        test_dataset = Cifar_Dataset(
            os.path.join(args.datapath, subpath), privtype, train=False, verbose=False)
        train_dataset = Cifar_Dataset( 
            os.path.join(args.datapath, subpath), privtype, train=True, verbose=False)
        train_x, train_y = train_dataset.img, train_dataset.gt
        priv_data = [None] * N_parties
        for cls_idx in range(N_class):
            idx = np.where(train_y == cls_idx)[0]
            totaln = idx.shape[0]
            idx_start = 0
            for i in range(N_parties):
                if i==N_parties-1:
                    cur_idx = idx[idx_start:]
                else:
                    idx_end = idx_start + int(split_arr[cls_idx][i]*totaln)
                    cur_idx = idx[idx_start: idx_end]
                    idx_start = idx_end
                if cur_idx == ():
                    continue
                if priv_data[i] is None:
                    priv_data[i] = {}
                    priv_data[i]['x'] = train_x[cur_idx]
                    priv_data[i]['y'] = train_y[cur_idx]
                    priv_data[i]['idx'] = cur_idx
                else:
                    priv_data[i]['idx'] = np.r_[(priv_data[i]['idx'], cur_idx)]
                    priv_data[i]['x'] = np.r_[(priv_data[i]['x'], train_x[cur_idx])]
                    priv_data[i]['y'] = np.r_[(priv_data[i]['y'], train_y[cur_idx])]
        all_priv_data = {}
        all_priv_data['x'] = train_x
        all_priv_data['y'] = train_y
    elif privtype=="pascal_voc2012":
        priv_data = [None] * N_parties
        for i in range(N_parties):
            priv_data[i] = {}
            path = pathlib.Path(args.datapath).joinpath('PASCAL_VOC_2012', 'dirichlet', f'alpha_{args.alpha:.0f}')
            party_img = np.load(path.joinpath(f'Party_{i}_X_data.npy'))
            party_label = np.load(path.joinpath(f'Party_{i}_y_data.npy'))
            priv_data[i]['x'] = party_img.copy()
            
            priv_data[i]['y'] = party_label.copy()
        
        print(f"y label : {priv_data[0]['y']}")
        for i in range(N_parties):
            print(f'Party_{i} data shape: {priv_data[i]["x"].shape}')
        print(f'Public data shape: {public_dataset.img.shape}')
        print(f'Test data shape: {test_dataset.img.shape}')
        train_dataset = None

    return priv_data, train_dataset, test_dataset, public_dataset, distill_loader

class Cifar_Dataset:
    def __init__(self, local_dir, data_type, train=True, with_coarse_label=False, verbose=False, distill=False, aug=True, public_percent=1):
        self.distill = distill
        if data_type == 'cifar10':
            if train == True:
                img, gt = [], []
                for i in range(1, 6):
                    file_name = None
                    file_name = os.path.join(local_dir + 'data_batch_{0}'.format(i))
                    X_tmp, y_tmp = (None, None)
                    with open(file_name, 'rb') as (fo):
                        datadict = pickle.load(fo, encoding='bytes')
                    X_tmp = datadict[b'data']
                    y_tmp = datadict[b'labels']
                    X_tmp = X_tmp.reshape(10000, 3, 32, 32)
                    y_tmp = np.array(y_tmp)
                    img.append(X_tmp)
                    gt.append(y_tmp)
                img = np.vstack(img)
                gt = np.hstack(gt)
            else:
                file_name = None
                file_name = os.path.join(local_dir + 'test_batch')
                with open(file_name, 'rb') as (fo):
                    datadict = pickle.load(fo, encoding='bytes')
                    img = datadict[b'data']
                    gt = datadict[b'labels']
                    img = img.reshape(10000, 3, 32, 32)
                    gt = np.array(gt)
        elif data_type == 'cifar100':
            if train == True:
                file_name = None
                file_name = os.path.abspath(local_dir + 'train')
                with open(file_name, 'rb') as (fo):
                    datadict = pickle.load(fo, encoding='bytes')
                    img = datadict[b'data']
                    if with_coarse_label:
                        gt = datadict[b'coarse_labels']
                    else:
                        gt = datadict[b'fine_labels']
                    img = img.reshape(50000, 3, 32, 32)
                    gt = np.array(gt)
            else:
                file_name = None
                file_name = os.path.join(local_dir + 'test')
                with open(file_name, 'rb') as (fo):
                    datadict = pickle.load(fo, encoding='bytes')
                    img = datadict[b'data']
                    if with_coarse_label:
                        gt = datadict[b'coarse_labels']
                    else:
                        gt = datadict[b'fine_labels']
                    # import ipdb; ipdb.set_trace()
                    img = img.reshape(10000, 3, 32, 32)
                    gt = np.array(gt)
        else:
            logging.info('Unknown Data type. Stopped!')
            return
        if verbose:
            logging.info(f'img shape: {img.shape}')
            logging.info(f'label shape: {gt.shape}')
        self.img = np.asarray(img)
        self.img = self.img.transpose((0, 2, 3, 1))
        self.gt = np.asarray(gt)
        total_N_img = img.shape[0]
        # import ipdb; ipdb.set_trace()
        if public_percent<1:
            total_N_img = int(total_N_img*public_percent)
            self.img = self.img[:total_N_img]
            self.gt = self.gt[:total_N_img]
            logging.info(f'Clip with {public_percent}, to {total_N_img}')
        self.fixid = np.arange(total_N_img)
        self.aug = aug
        self.train = train
        
        self.train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            utils.Cutout(16),
            ])
        
        self.test_transformer = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, idx):
        image = self.img[idx]
        label = self.gt[idx]
        fixid = self.fixid[idx]
        # transimage = Image.fromarray(image.transpose(1,2,0).astype('uint8'))
        # transimage = transforms.ToPILImage()(transimage)
        transimage = Image.fromarray(image)
        
        if self.train and self.aug:
            transformer = self.train_transformer 
        else:
            transformer = self.test_transformer
        transimage = transformer(transimage)
        if self.distill:
            return (transimage, label, idx)
        else:
            return (transimage, label, fixid)

class Dataset_fromarray(Cifar_Dataset):
    def __init__(self, img_array, gt_array, train=True, verbose=False, multitrans=1, distill=False, aug=True):
        self.distill = distill
        self.img = img_array
        self.gt = gt_array
        self.fixid = np.arange(self.img.shape[0])
        self.multitrans = multitrans
        self.train = train
        
        self.train_transformer = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                utils.Cutout(16),
                ])
        
        self.test_transformer = transforms.Compose([
                 transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])
        # self.transformer2= transforms.Compose(self.transformer.transforms[:-1]) #if multi, no cutout
        if verbose == True:
            logging.info(f'img shape: {self.img.shape}')
            logging.info(f'label shape: {self.gt.shape}')
        self.aug = aug

