from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import unittest
import numpy as np
import torch.optim as optim

class MHALoss(torch.nn.Module):
    '''
    summary : FedMHAD (Multihead Attention loss)
    '''
    def __init__(self, distill_heads = 3):
        super(MHALoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-8)
        self.distill_heads = distill_heads
        
    # def forward(self, client_attentions, central_attention, weight):
    #     '''
    #     args :
    #         client_attentions : (num_clients, batch_size, num_heads, img_height, img_width)
    #         central_attention : (batch_size, num_heads, img_height, img_width)
    #         weight : (num_clients)
    #     returns :
    #         weighted_loss : (1)
    #     '''
        
    #     if weight is not None:
    #         self.weight = torch.tensor(weight, dtype=torch.float)
    #     else:
    #         self.weight = torch.ones(self.num_clients, dtype=torch.float)
    #     client_attentions = client_attentions[:, :, 0:self.distill_heads, :, :]
    #     central_attention = central_attention[:, 0:self.distill_heads, :, :]
    #     # assert len(clients_output) == self.num_clients, "Number of clients_output should match num_clients"
    #     num_clients = client_attentions.shape[0]
    #     loss = 0
    #     for i in range(num_clients):
    #         cosine_sim = self.cosine_similarity(client_attentions[i].contiguous().view(-1), central_attention.contiguous().view(-1))
    #         loss += self.weight[i] * (1 - cosine_sim).mean()
    #         print('cosine_sim : ', cosine_sim)
    #     loss /= num_clients
    #     print('MHA Loss : ', loss)
    #     # print('MHA Loss : ', loss)
    #     return loss
    def forward(self, inter_input, union_input, target):
        # inter_input : ensembled gradcam image (intersection)
        # union_input : ensembled gradcam image (union)
        # target : central gradcam image
        p1, b1 = 10, 0.6
        p2, b2 = 10, 0.3
        
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)
        if not isinstance(inter_input, torch.Tensor):
            inter_input = torch.tensor(inter_input)
            
        t_A = torch.sigmoid(-p1*(target-b1))
        # Weighted Average sum
        loss1 = - torch.sum(torch.dot(t_A.view(-1), inter_input.view(-1)))/torch.sum(t_A)
        t_U = torch.sigmoid(-p2*(union_input-b2))
        loss2 = - torch.sum(torch.dot(t_U.view(-1), target.view(-1)))/torch.sum(target)
        print('intersection loss : ', loss1, 'union loss : ', loss2)
        return loss1 + loss2
    
    def get_union_intersection_image(self, client_attentions):
        '''
         client_attentions : (num_clients, batch_size, num_heads, img_height, img_width)
        '''
        if not isinstance(client_attentions, torch.Tensor):
            client_attentions = torch.tensor(client_attentions)
        
        union = torch.max(client_attentions, dim=0)[0]
        intersection = torch.min(client_attentions, dim=0)[0]
        print('union : ', union.shape) # (batch_size, num_heads, img_height, img_width)
        return union, intersection

class TestMHALoss(unittest.TestCase):
     
    def test_compute_attention_similarity_losses(self):
        n_clients = 3
        batch_size = 4
        n_heads = 8
        img_height = 32
        img_width = 32
        
        client_attentions = torch.randn(n_clients, batch_size, n_heads, img_height, img_width, requires_grad=True)
        central_attention = torch.randn(batch_size, n_heads, img_height, img_width, requires_grad=True)
        weight = [0.2, 0.2, 0.2, 0.2, 0.2]  # Example weight for each client

        cosine_similarity_loss = MHALoss().requires_grad_(True)
        loss = cosine_similarity_loss(client_attentions, central_attention, weight)
        print(loss.grad_fn)
        print(loss.requires_grad)
        print(loss.grad)
        
        print(loss)
        client_attentions = torch.randn(n_clients, batch_size, n_heads, img_height, img_width, requires_grad=True)
        central_attention = torch.randn(batch_size, n_heads, img_height, img_width, requires_grad=True)
        loss = cosine_similarity_loss(client_attentions, central_attention, weight)
        print(loss.grad_fn)
        print(loss.requires_grad)
        print(loss.grad)
        
        print(loss)
    # def test_compute_weighted_loss(self):
    #     # Example usage
    #     global_model_output = torch.randn(32, 10)
    #     clients_output = [torch.randn(32, 10) for _ in range(5)]
    #     weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Example weights for each client

    #     cosine_similarity_loss = CosineSimilarityGradientLoss(num_clients=5, weight=weights)
    #     loss = cosine_similarity_loss(global_model_output, clients_output)
    #     print(loss)
    #     print(loss.grad_fn)
    #     print(loss.requires_grad)
    #     print(loss.grad)
        
        
    # def test_compute_weighted_loss(self):
    #     client_losses = torch.randn(3)
    #     client_weight = torch.randn(3)
    #     mha_loss = MHALoss()
    #     loss = mha_loss.compute_weighted_loss(client_losses, client_weight)
    #     self.assertEqual(loss.shape, (1,))
        
    # def test_mha_loss(self):
    #     client_attentions = torch.randn(3, 4, 8, 32, 32)
    #     central_attention = torch.randn(4, 8, 32, 32)
    #     mha_loss = MHALoss()
    #     loss = mha_loss(client_attentions, central_attention)
    #     self.assertEqual(loss.shape, (1,))
        
if __name__ == '__main__':
    unittest.main()