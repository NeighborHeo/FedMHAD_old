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
from .objectives import GradientLoss

class MHALoss(torch.nn.Module):
    '''
    summary : FedMHAD (Multihead Attention loss)
    '''
    def __init__(self):
        super(MHALoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-8)
        
    def forward(self, client_attentions, central_attention, weight):
        '''
        args :
            client_attentions : (num_clients, batch_size, num_heads, img_height, img_width)
            central_attention : (batch_size, num_heads, img_height, img_width)
            weight : (num_clients)
        returns :
            weighted_loss : (1)
        '''
        
        if weight is not None:
            self.weight = torch.tensor(weight, dtype=torch.float)
        else:
            self.weight = torch.ones(self.num_clients, dtype=torch.float)
        # mha_images = mha_images[:, :, 0:2, :, :]
        # target = target[:, 0:2, :, :]
        # assert len(clients_output) == self.num_clients, "Number of clients_output should match num_clients"
        num_clients = client_attentions.shape[0]
        loss = 0
        for i in range(num_clients):
            cosine_sim = self.cosine_similarity(client_attentions[i].contiguous().view(-1), central_attention.contiguous().view(-1))
            loss += self.weight[i] * (1 - cosine_sim).mean()
        loss /= num_clients
        print('MHA Loss : ', loss)
        return loss

class CosineSimilarityGradientLoss(GradientLoss):
    def __init__(self, scale=1.0, task_regularization=0.0, num_clients=5, weight=None, **kwargs):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization
        self.num_clients = num_clients
        if weight is not None:
            self.weight = torch.tensor(weight, dtype=torch.float)
        else:
            self.weight = torch.ones(self.num_clients, dtype=torch.float)

    def gradient_based_loss(self, gradient_rec, gradient_data):
        loss = 0
        for i, (rec, data) in enumerate(zip(gradient_rec, gradient_data)):
            cosine_sim = F.cosine_similarity(rec, data, dim=1, eps=1e-8)
            loss += self.weight[i] * (1 - cosine_sim).mean()
        loss /= self.num_clients
        return loss * self.scale

    def __repr__(self):
        return f"Cosine Similarity Gradient Loss with scale={self.scale}, num_clients={self.num_clients}, and task reg={self.task_regularization}"
        
    # def forward(self, mha_images, target, weight=None):
    #     sim_loss = self.compute_attention_similarity_losses(mha_images, target)
    #     weighted_loss = self.compute_weighted_loss(sim_loss, weight)
    #     return weighted_loss
    
    # def compute_attention_similarity_losses(self, client_attentions, central_attention):
    #     """
    #     Args:
    #         client_attentions (torch.Tensor): (num_clients, batch_size, num_heads, img_height, img_width)
    #         central_attention (torch.Tensor): (batch_size, num_heads, img_height, img_width)
    #     Returns:
    #         similarity_losses (torch.Tensor): (num_clients)
    #     """
    #     similarity_losses = []
    #     cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    #     for i in range(client_attentions.shape[0]):
    #         # print('Client Attention : ', client_attentions[i].shape, 'central_attention : ', central_attention.shape)
    #         cs_value = cosine_similarity(client_attentions[i].reshape(-1), central_attention.reshape(-1))
    #         # print('Cosine Similarity : ', cs_value)
    #         attention_loss = torch.tensor(1) - cs_value
    #         # print('Attention Loss : ', attention_loss)
    #         similarity_losses.append(attention_loss)

    #     return torch.stack(similarity_losses)

    # def compute_weighted_loss(self, client_losses, client_weight):
    #     """
    #     Args:
    #         client_losses (torch.Tensor): (num_clients)
    #         client_weight (torch.Tensor): (num_clients)
    #     Returns:
    #         client_weighted_loss (torch.Tensor): (1)
    #     """
    #     if client_weight is None:
    #         client_weighted_loss = torch.mean(client_losses)
    #     else:    
    #         weighted_losses = client_losses * client_weight
    #         client_weighted_loss = torch.sum(weighted_losses)
            
    #     return client_weighted_loss

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