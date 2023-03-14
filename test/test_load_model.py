# %%
import sys
sys.path.append('..')
import utils
import os
import pathlib
import argparse
from tensorboardX import SummaryWriter
import logging
from datetime import datetime
import torch 
import mymodels 
import mydataset 
from torch.utils.data import DataLoader
from utils.myfed import *
import yaml
# %%

yamlfilepath = pathlib.Path.cwd().parent.joinpath('config.yaml')
args = yaml.load(yamlfilepath.open('r'), Loader=yaml.FullLoader)
args = argparse.Namespace(**args)
args.datapath = "~/.data"
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

# 1. data
args.datapath = os.path.expanduser(args.datapath)

if args.dataset == 'cifar10':
    publicdata = 'cifar100'
    args.N_class = 10
elif args.dataset == 'cifar100':
    publicdata = 'imagenet'
    args.N_class = 100
elif args.dataset == 'pascal_voc2012':
    publicdata = 'mscoco'
    args.N_class = 20

assert args.dataset in ['cifar10', 'cifar100', 'pascal_voc2012']

priv_data, _, test_dataset, public_dataset, distill_loader = mydataset.data_cifar.dirichlet_datasplit(
    args, privtype=args.dataset, publictype=publicdata, N_parties=args.N_parties, online=not args.oneshot, public_percent=args.public_percent)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, sampler=None)
val_loader = DataLoader(
    dataset=public_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, sampler=None)

net = mymodels.define_model(modelname=args.model_name, num_classes=args.N_class)
net 
n = 0
loadname = os.path.join("/home/suncheol/code/VFL/FedMAD/checkpoints_backup/pascal_voc2012/a1.0+sd1+e300+b16+lkl", str(n)+'.pt')
if os.path.exists(loadname):
    localmodels = torch.load(loadname)
    #self.localmodels[n].load_state_dict(self.best_statdict, strict=True)
    logging.info(f'Loading Local{n}......')
    print('filepath : ', loadname)
    utils.load_dict(loadname, net)
# loadname = os.path.join("/home/suncheol/code/VFL/FedMAD/checkpoints/pascal_voc2012/a1.0+sd1+e300+b16+lkl/model-0.pth")
loadname = os.path.join("/home/suncheol/code/FedTest/pytorch-models/checkpoint/pascal_voc_vit_tiny_patch16_224_0.0001_-1/ckpt.pth")
if os.path.exists(loadname):
    localmodels = torch.load(loadname)
    #self.localmodels[n].load_state_dict(self.best_statdict, strict=True)
    logging.info(f'Loading Local{n}......')
    print('filepath : ', loadname)
    utils.load_dict(loadname, net)
import copy
models = []
for i in range(0, 5):
    model = copy.deepcopy(net)
    # loadname = os.path.join(f"/home/suncheol/code/VFL/FedMAD/checkpoints/pascal_voc2012/a1.0+sd1+e300+b16+lkl/model-{i}.pth")
    loadname = os.path.join(f"/home/suncheol/code/FedTest/pytorch-models/checkpoint/pascal_voc_vit_tiny_patch16_224_0.0001_{i}/ckpt.pth")
    if os.path.exists(loadname):
        localmodels = torch.load(loadname)
        #self.localmodels[n].load_state_dict(self.best_statdict, strict=True)
        logging.info(f'Loading Local{n}......', 'filepath : ', loadname)
        utils.load_dict(loadname, model)
    models.append(model)
len(models)
# show 1 batch of data
import matplotlib.pyplot as plt
import numpy as np
import torchvision
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(test_loader)
images, labels, _ = dataiter.next()

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
# multi label to multi captions
def multi_label_to_multi_captions(labels):
    captions = []
    for label in labels:
        caption = []
        for i in range(len(label)):
            if label[i] == 1:
                caption.append(VOC_CLASSES[i])
        captions.append(caption)
    return captions
labels = multi_label_to_multi_captions(labels)
grad_cam_images = []
pred_labels = []
for model in models:
    grad_cam_images.append(model.module.get_class_activation_map(images, labels))
    m = torch.nn.Sigmoid()
    th = 0.3
    outputs = m(model(images)).detach().cpu().numpy()
    outputs[outputs > th] = 1
    outputs[outputs <= th] = 0
    pred = multi_label_to_multi_captions(outputs)
    pred_labels.append(pred)
# grayscale_cam = net.module.get_class_activation_map(images, labels)

grad_cam_images = torch.stack([torch.tensor(grad_cam_images[i]) for i in range(len(grad_cam_images))])
grad_cam_images.shape # n_clients * b * 224 * 224
union_cam = torch.max(grad_cam_images, dim=0)[0]
intersection_cam = torch.min(grad_cam_images, dim=0)[0]
union_cam.cpu().shape

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
# multi label to multi captions
def multi_label_to_multi_captions(labels):
    captions = []
    for label in labels:
        caption = []
        for i in range(len(label)):
            if label[i] == 1:
                caption.append(VOC_CLASSES[i])
        captions.append(caption)
    return captions
labels = multi_label_to_multi_captions(labels)
row = 10
col = 8
import matplotlib.pyplot as plt
plt.figure(figsize=(3 * col, 3 * row))
for j in range(0, row):
    plt.subplot(row, col, j*col+1)
    plt.imshow(images[j].cpu().permute(1, 2, 0))
    plt.title(f'GT: {labels[j]}')
    for i in range(0, 5):
        plt.subplot(row, col, j*col+i+2)
        plt.imshow(grad_cam_images[i].cpu()[j])
        plt.title(f'client{i+1}: {pred_labels[i][j]}')
    plt.subplot(row, col, j*col+7)
    plt.imshow(union_cam.cpu()[j])
    plt.title('union')
    plt.subplot(row, col, j*col+8)
    plt.imshow(intersection_cam.cpu()[j])
    plt.title('intersection')
     
plt.show()
# plt.tight_layout()
# grayscale_cam # b * 224 * 224
# grayscale_cam = torch.tensor(grayscale_cam)
# # n_clients * b * 224 * 224
# grayscale_cam = torch.stack([grayscale_cam, grayscale_cam], dim=0)

# grayscale_cam.shape
# grayscale_cam is batch_size x 224 x 224
# union is maximum of all CAMs 
# intersection is minimum of all CAMs
union_cam = torch.max(torch.tensor(grayscale_cam), dim=0)[0]
intersection_cam = torch.min(torch.tensor(grayscale_cam), dim=0)[0]
union_cam.numpy().shape
# plt.imshow(mha[0, :, :, :].cpu().detach().numpy())
# image grid 
img = mha[0, :, :, :].cpu().detach().numpy()
 
imshow(img)
 

mha_images = [] 
th_images = []
for model in models:
    mha, th = model.module.get_attention_maps_postprocessing_(images.cuda())
    mha_images.append(mha)
    th_images.append(th)
    
mha_images = torch.stack([torch.tensor(mha_images[i]) for i in range(len(mha_images))])
th_images = torch.stack([torch.tensor(th_images[i]) for i in range(len(th_images))])
mha_images.shape # n_clients * b * 224 * 224
th_images.shape # n_clients * b * 224 * 224
print(mha_images.shape, th_images.shape)


n_clients, n_images, n_head, h, w = mha_images.shape
row = 4 * n_head
col = 1 + n_clients + 2
import matplotlib.pyplot as plt
plt.figure(figsize=(3 * col, 3 * row))
for j in range(0, row):
    _j = j // n_head
    if j % 3 == 0:
        plt.subplot(row, col, j*col+1)
        plt.imshow(images[_j].numpy().transpose(1, 2, 0))
        plt.title(f'original')
    for i in range(0, 5):
        k = j % n_head
        plt.subplot(row, col, j*col+i+2)
        # print(j*col+i+2, i, j, k)
        plt.imshow(mha_images[i, _j, k, :, :].numpy())
        plt.title(f'client{i+1}, image{_j}, head{k}')
plt.show()
plt.tight_layout()
central_model = copy.deepcopy(net)
loadname = os.path.join(f"/home/suncheol/code/VFL/FedMAD/checkpoints/pascal_voc2012/a1.0+sd1+e300+b16+lkl+slmha/oneshot_c1_q0.0_n0.0/q0.0_n0.0_ADAM_b128_0.001_100_1e-05_m0.9_e53_0.41.pt")
if os.path.exists(loadname):
    localmodels = torch.load(loadname)
    #self.localmodels[n].load_state_dict(self.best_statdict, strict=True)
    logging.info(f'Loading Local{n}......', 'filepath : ', loadname)
    utils.load_dict(loadname, central_model)
# models.append(central_model)
loss_function = Loss.mha_loss()
# loss_function = loss_function.
mha, th = central_model.module.get_attention_maps_postprocessing_(images.cuda())
# mha_images = []
# for model in models:
#     mha = model.module.get_attention_maps(images.cuda())[-1]
#     # mha = model.module.get_attention_maps(images.cuda())[-1]
#     # print("mha shape : ", mha.shape, "thrs shape : ", thrs.shape)
#     print("mha shape : ", mha.shape)
#     mha_images.append(mha)

# mha_images = torch.stack([torch.tensor(mha_images[i]) for i in range(len(mha_images))])
# mha_images.shape # n_clients * b * 224 * 224
# print(mha.shape)

# w, h = images.shape[2] // 16, images.shape[3] // 16
# nImg, nh, _, _  = mha.shape
# mha = mha[:, :, 0, 1:].reshape(nImg, nh, -1)
# # mha = mha.view(-1, w, h)
#     # mha = mha.reshape(nh, w_featmap, h_featmap)
#     # mha = nn.functional.interpolate(mha.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
# mha = mha.reshape(nImg, nh, w, h)

# mha = nn.functional.interpolate(mha, scale_factor=16, mode="nearest")
# mha.shape


# th[i, 0, :, :] .unique()
unique, counts = torch.unique(th_images, return_counts=True)
dict(zip(unique, counts))
def get_attention_maps_postprocessing(self, x):
    # x.shape = (batch, channel, height, width) = 16, 3, 197, 197
    mha = self.get_attention_maps(x)[-1]
    
    # mha.shape = (batch, head, height, width) = 16, 3, 197, 197
    w_featmap = x.shape[-2] // 16
    h_featmap = x.shape[-1] // 16
    nh = mha.shape[1] # number of head
    # w_featmap = 14, h_featmap = 14, nh = 3
    # we keep only the output patch attention
    mha = mha[0, :, 0, 1:].reshape(nh, -1)
    # mha.shape = 3, 196(14*14)
    # we keep only a certain percentage of the mass
    val, idx = torch.sort(mha)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - 0.6)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
    mha = mha.reshape(nh, w_featmap, h_featmap)
    mha = nn.functional.interpolate(mha.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
    return mha, th_attn


mh, th = get_attention_maps_postprocessing(models[0].module, images.cuda())
mh.shape, th.shape
images[0][0].shape
from utils.visualize import * 
pic_attn_color = show_attn_color(images[0].permute(1, 2, 0).cpu().numpy().unsequeeze(0), mha, th, head=[0,1,2])
imshow(pic_attn_color)
import matplotlib.pyplot as plt
import numpy as np
import torchvision
# plot mha
for i in range(0, 16):
    plt.subplot(4, 4, i+1)
    # plt.imshow(mha[i, 0, :, :])
    plt.imshow(th[i, 2, :, :])
print(images.shape)
mha_images = []
for model in models:
    mha, thrs = model.module.get_attention_maps_postprocessing(images.cuda())
    # mha = model.module.get_attention_maps(images.cuda())[-1]
    # print("mha shape : ", mha.shape, "thrs shape : ", thrs.shape)
    print("mha shape : ", mha.shape)
    mha_images.append(mha)

print(len(mha_images))
mha_images = torch.stack([torch.tensor(mha_images[i]) for i in range(len(mha_images))])
print(mha_images.shape)
# imshow 

# grid = torchvision.utils.make_grid(mha_images[0])
torchvision.utils.make_grid(mha_images[0])
mha_images.shape
3* 197
# mha_images = mha_images.reshape(5, 16, 591, 197).cpu().detach().numpy()
# grayscale_cam is batch_size x 224 x 224
# union is maximum of all CAMs 
# intersection is minimum of all CAMs

mha_images.reshape(5, 3, )
union_cam = torch.max(torch.tensor(mha_images), dim=0)[0]
intersection_cam = torch.min(torch.tensor(mha_images), dim=0)[0]
union_cam.numpy().shape
row = 1
col = 8
import matplotlib.pyplot as plt
plt.figure(figsize=(3 * col, 3 * row))
for j in range(0, row):
    plt.subplot(row, col, j*col+1)
    plt.imshow(images[j].numpy().transpose(1, 2, 0))
    plt.title(f'original')
    for i in range(0, 5):
        plt.subplot(row, col, j*col+i+2)
        plt.imshow(mha_images[i].numpy()[j])
        plt.title(f'client{i}')
    plt.subplot(row, col, j*col+7)
    plt.imshow(union_cam.numpy()[j])
    plt.title('union')
    plt.subplot(row, col, j*col+8)
    plt.imshow(intersection_cam.numpy()[j])
    plt.title('intersection')
plt.show()
plt.tight_layout()
union_cam = torch.max(mha_images, dim=0)[0]
intersection_cam = torch.min(mha_images, dim=0)[0]
union_cam.numpy().shape

row = 4
col = 8
import matplotlib.pyplot as plt
plt.figure(figsize=(3 * col, 3 * row))
for j in range(0, row):
    plt.subplot(row, col, j*col+1)
    plt.imshow(images[j].numpy().transpose(1, 2, 0))
    plt.title(f'original')
    for i in range(0, 5):
        plt.subplot(row, col, j*col+i+2)
        plt.imshow(grad_cam_images[i].numpy()[j])
        plt.title(f'client{i}')
    plt.subplot(row, col, j*col+7)
    plt.imshow(union_cam.numpy()[j])
    plt.title('union')
    plt.subplot(row, col, j*col+8)
    plt.imshow(intersection_cam.numpy()[j])
    plt.title('intersection')
plt.show()
plt.tight_layout()
# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class at_loss(torch.nn.Module):
#     '''
#     summary : FedAD attention loss function
#     '''
#     def __init__(self): #, T=3, singlelabel=False
#         super().__init__()
#         # self.T = T
#         # self.singlelabel = singlelabel
#         # self.criterion= torch.nn.KLDivLoss(reduction='batchmean')

#     def forward(self, inter_input, union_input, target):
#         # inter_input : ensembled gradcam image (intersection)
#         # union_input : ensembled gradcam image (union)
#         # target : central gradcam image
#         p1, b1 = 10, 0.6
#         p2, b2 = 10, 0.3
#         t_A = torch.sigmoid(-p1*(target-b1))
#         # Weighted Average sum
#         loss1 = - torch.sum(torch.dot(t_A.view(-1), inter_input.view(-1)))/torch.sum(t_A)
#         t_U = torch.sigmoid(-p2*(union_input-b2))
#         loss2 = - torch.sum(torch.dot(t_U.view(-1), target.view(-1)))/torch.sum(target)
#         print('intersection loss : ', loss1, 'union loss : ', loss2)
#         return loss1 + loss2


# def weight_gradcam(cam_images, countN):#nlcoal*batch*nclass
#     #softLogits = torch.nn.Softmax(dim=2)(logits)
#     # cam_images = n_clinets * batch size * image width * image height
#     # union is maximum of all clients cam_images = batch size * image width * image height
#     union = torch.max(torch.tensor(cam_images.clone()), dim=0)[0]
#     # intersection is minimum of all clients cam_images = batch size * image width * image height
#     intersection = torch.min(torch.tensor(cam_images.clone()), dim=0)[0]
#     return union, intersection
# at_loss = at_loss()
# union_cam, intersection_cam = weight_gradcam(grayscale_cam, 2)
# at_loss(intersection_cam, union_cam, grayscale_cam[0])
# $$ T(\bm A) = \frac {1}{1+exp(-\rho (\bm A-b))}. (7) $$

# $$ \label {eqinter} \Loss _\text {inter}({\mathbf {\widetilde A}}, {\mathbf I}) = - \frac {1}{C} \sum _c{{\frac {\sum _{hw} {I_{hw}^{c} \cdot T(\widetilde {A}_{hw}^c; \rho _1, b_1)}}{\sum _{hw} {I_{hw}^{c}}}}}, (8) $$

# $$ \label {equnion} \Loss _\text {union}({\mathbf {\widetilde A}}, {\mathbf U}) = - \frac {1}{C} \sum _c{ {\frac {\sum _{hw} {\widetilde {A}_{hw}^c \cdot T(U_{hw}^c; \rho _2, b_2)}}{\sum _{hw} {\widetilde {A}_{hw}^c}} }},
# (9) $$
# plt.imshow(union_cam)
# plt.imshow(intersection_cam)
# grayscale_cam[0].shape
# %matplotlib inline
# plt.imshow(grayscale_cam[2])
# plt.show()
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
img_list = []
plt.figure(figsize=(10, 10))
for i in range(args.batchsize):
    np_input = images[i].cpu().numpy()
    np_input = np.transpose(np_input, (1, 2, 0))
    np_input.shape
    grayscale_cam_ = grayscale_cam[i]
    cam_image = show_cam_on_image(np_input, grayscale_cam_, use_rgb=True)
    img_list.append(cam_image)
    plt.subplot(4, 4, i+1)
    # plt.imshow(grayscale_cam_)
    plt.imshow(cam_image)
# show images and labels
imshow(torchvision.utils.make_grid(images))
# print labels

from torchvision import transforms as pth_transforms
# from visualize_attention import company_colors, apply_mask2
from PIL import Image, ImageDraw
from utils.visualize import * 
# read image 

def show_attn(net, img, index=None, nlayer=0):
    w_featmap = img.shape[-2] // 16
    h_featmap = img.shape[-1] // 16

    # attentions = vit.get_last_selfattention(img.cuda())
    # attentions = net.module.get_attention_maps(img.cuda())[-1]
    attentions = net.module.get_attention_maps(img.cuda())[nlayer]

    print('attentions shape', attentions.shape)
    print('attentions', attentions)
    nh = attentions.shape[1] # number of head
    print('number of head', nh)
    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - 0.6)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
    print('th_attn.shape: ', th_attn.shape)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    prefix = f'id{index}_' if index is not None else ''
    os.makedirs('pics/', exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join('pics/', "img" + ".png"))
    img = Image.open(os.path.join('pics/', "img" + ".png"))

    attns = Image.new('RGB', (attentions.shape[2] * nh, attentions.shape[1]))
    for j in range(nh):
        print('attentions[j].shape: ', attentions[j].shape)
        fname = os.path.join('pics/', "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        attns.paste(Image.open(fname), (j * attentions.shape[2], 0))

    return attentions, th_attn, img, attns


# img = Image.open('../data/NIH/processed/images_001/images/00000001_000.png')
# img = img.resize((224, 224))
# img = images.permute(0, 2, 3, 1)[4]
# img = torch.tensor(np.array(img)).permute(2, 0, 1)
img = images[4]


def show_attention(net, img, nlayer=-1):
    img = torch.tensor(np.array(img))
    img.shape
    transform = pth_transforms.Compose([
        pth_transforms.ToPILImage(),
        pth_transforms.Grayscale(num_output_channels=3),
        pth_transforms.Resize([224, 224]),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    print(img.shape)
    # make the image divisible by the patch size
    w, h = img.shape[-2] - img.shape[-2] % 16, img.shape[-1] - img.shape[-1] % 16
    print(w, h)
    img = img[:, :w, :h].unsqueeze(0)
    print(img.shape)
    attentions, th_attn, pic_i, pic_attn = show_attn(net, img, nlayer=-1)
    print("attentions.shape: ", attentions.shape)
    print("th_attn.shape: ", th_attn.shape)
    print("pic_i.shape: ", pic_i.size)
    pic_attn_color = show_attn_color(img[0].permute(1, 2, 0).cpu().numpy(), attentions, th_attn, head=[0,1,2])
    final_pic = Image.new('RGB', (pic_i.size[1] * 2 + pic_attn.size[0], pic_i.size[1]))
    final_pic.paste(pic_i, (0, 0))
    final_pic.paste(pic_attn_color, (pic_i.size[1], 0))
    final_pic.paste(pic_attn, (pic_i.size[1] * 2, 0))
    display(final_pic)
    return attentions

attentions = []
for model in models:
    attention = show_attention(model, images[4])
    attentions.append(attention)

import loss as Loss
# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class mha_loss(torch.nn.Module):
    '''
    summary : FedMHAD (Multihead Attention loss)
    '''
    def __init__(self): #, T=3, singlelabel=False
        super().__init__()
        # self.T = T
        # self.singlelabel = singlelabel
        # self.criterion= torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, inter_input, union_input, target):
        # inter_input : ensembled gradcam image (intersection)
        # union_input : ensembled gradcam image (union)
        # target : central gradcam image
        p1, b1 = 1, 0.3
        p2, b2 = 1, 0.4
        target = torch.tensor(target)
        t_A = torch.sigmoid(-p1*(target-b1))
        # Weighted Average sum
        loss1 = -torch.sum(torch.dot(t_A.view(-1), inter_input.view(-1)))/torch.sum(t_A)
        t_U = torch.sigmoid(-p2*(union_input-b2))
        loss2 = -torch.sum(torch.dot(t_U.view(-1), target.view(-1)))/torch.sum(target)
        print('intersection loss : ', loss1, 'union loss : ', loss2)
        return loss1 + loss2


def weight_multihead_attention_map(mha_images, countN):#nlcoal*batch*nclass
    # mha_images = n_clinets * batch size * n_head * image width * image height
    # union is maximum of all clients 
    # union = batch size * n_head * image width * image height
    union = torch.max(torch.tensor(mha_images), dim=0)[0]

    # intersection is minimum of all clients 
    # intersection = batch size * n_head * image width * image height
    intersection = torch.min(torch.tensor(mha_images), dim=0)[0]
    print('mha_images : ', torch.tensor(mha_images).shape, 'union : ', union.shape, 'intersection : ', intersection.shape)
    return union, intersection
attentions = np.array(attentions)
attentions1 = attentions[0, :, :, :]
attentions2 = attentions[1, :, :, :]
attentions3 = attentions[2, :, :, :]
attentions4 = attentions[3, :, :, :]

tn_attentions1 = torch.tensor(attentions1)
tn_attentions2 = torch.tensor(attentions2)
tn_attentions3 = torch.tensor(attentions3)
tn_attentions4 = torch.tensor(attentions4)

loss = mha_loss()
print('loss: ', loss(tn_attentions3, tn_attentions4, tn_attentions1))
print('loss: ', loss(tn_attentions4, tn_attentions3, tn_attentions2))
len(net.module.get_attention_maps(img.cuda()))
out = net(img.cuda())
for i in range(12):
    attentions = net.module.get_attention_maps(img.cuda())[i]
    np_mean = np.mean(attentions.cpu().numpy())
    np_std = np.std(attentions.cpu().numpy())
    for j in range(attentions.shape[1]):
        np_mean = np.mean(attentions[:, j, :, :].cpu().numpy())
        np_std = np.std(attentions[:, j, :, :].cpu().numpy())
        print(f'layer {i} head {j} mean: {np_mean}, std: {np_std}')
    # print(f'layer {i} mean: {np_mean}, std: {np_std}')
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
img_list = []
plt.figure(figsize=(10, 10))
for i in range(args.batchsize):
    np_input = images[i].cpu().numpy()
    np_input = np.transpose(np_input, (1, 2, 0))
    np_input.shape
    grayscale_cam_ = grayscale_cam[i]
    cam_image = show_cam_on_image(np_input, grayscale_cam_, use_rgb=True)
    img_list.append(cam_image)
    plt.subplot(4, 4, i+1)
    plt.imshow(grayscale_cam_)
    plt.axis('off')
plt.show()


# %%
