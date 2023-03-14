from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
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

    # def forward(self, inter_input, union_input, target):
    #     # inter_input : ensembled gradcam image (intersection)
    #     # union_input : ensembled gradcam image (union)
    #     # target : central gradcam image
    #     similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #     loss1 = 1-similarity(inter_input, target).mean()
    #     loss2 = 1-similarity(union_input, target).mean()
    #     # print('intersection loss : ', loss1.shape, 'union loss : ', loss2.shape)
    #     print('intersection loss : ', loss1, 'union loss : ', loss2)
    #     return loss1 + loss2

    def forward(self, mha_images, target, weights=None):
        # inter_input : ensembled gradcam image (intersection)
        # union_input : ensembled gradcam image (union)
        # target : central gradcam image
        similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        losses = []
        for i in range(mha_images.shape[0]):
            loss = 1-similarity(mha_images[i], target).mean()
            losses.append(loss)
        # if weights is not None:
        #     weighted_loss = torch.mean(torch.stack(losses) * weights)
        # else:
        weighted_loss = torch.mean(torch.stack(losses))
        return weighted_loss


    # def forward(self, inter_input, union_input, target):
    #     # inter_input : ensembled gradcam image (intersection)
    #     # union_input : ensembled gradcam image (union)
    #     # target : central gradcam image
    #     p1, b1 = 10, 0.6
    #     p2, b2 = 10, 0.3
    #     target = torch.tensor(target)
    #     t_A = torch.sigmoid(-p1*(target-b1))
    #     # Weighted Average sum
    #     print('inter_input : ', inter_input.shape, 'target : ', target.shape)
    #     loss1 = -torch.sum(torch.dot(t_A.view(-1), inter_input.view(-1)))/torch.sum(t_A)
    #     t_U = torch.sigmoid(-p2*(union_input-b2))
    #     loss2 = -torch.sum(torch.dot(t_U.view(-1), target.view(-1)))/torch.sum(target)
    #     print('intersection loss : ', loss1, 'union loss : ', loss2)
    #     return loss1 + loss2


def weight_multihead_attention_map(mha_images, countN):#nlcoal*batch*nclass
    # mha_images = n_clinets * batch size * n_head * image width * image height
    # union is maximum of all clients 
    # union = batch size * n_head * image width * image height
    union = torch.max(mha_images, dim=0)[0]

    # intersection is minimum of all clients 
    # intersection = batch size * n_head * image width * image height
    intersection = torch.min(mha_images, dim=0)[0]
    print('mha_images : ', mha_images.shape, 'union : ', union.shape, 'intersection : ', intersection.shape)
    return union, intersection