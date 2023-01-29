from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# class mha_loss(torch.nn.modules):
#     '''
#     summary : kullback-leibler divergence loss      
#     '''
#     def __init__(self, T=3, singlelabel=False):
#         super().__init__()
#         self.T = T
#         self.singlelabel = singlelabel
#         self.criterion= torch.nn.KLDivLoss(reduction='batchmean')

#     def forward(self, input, target):
#         if self.singlelabel:
#             soft_in = torch.nn.functional.softmax(input/self.T, dim=1)
#             soft_target = torch.nn.functional.softmax(target/self.T, dim=1).float()
#             loss = self.criterion(soft_in.log(), soft_target)
#         else:
#             soft_in = torch.nn.functional.sigmoid(input/self.T)
#             soft_target = torch.nn.functional.sigmoid(target/self.T)
#             loss = self.criterion(soft_in.log(), soft_target)
#         return self.T*self.T*loss

def weight_multiheadattention(logits, countN, noweight=False, clscount=False, votethresh=0, singlabel=True):#nlcoal*batch*nclass
    #softLogits = torch.nn.Softmax(dim=2)(logits)
    nlocal = logits.shape[0]
    nbatch = logits.shape[1]
    ncls = logits.shape[2]
    labels = logits.argmax(axis=2)#nlcoal*batch
    votes = torch.zeros((logits.shape[1:]))#nbatch*nclass
    for clsn in range(logits.shape[2]):
        votes[:,clsn] = (labels==clsn).sum(dim=0)
    psedolabel = votes.argmax(axis=1)#nbatch
    psedolabel = psedolabel.expand((nlocal,nbatch)).cuda()
    votemask = labels==psedolabel#nlcoal*batch #delete if <10?
    if votethresh:
        votesum = votemask.sum(dim=0)#nbatch
        votePass = (votesum>votethresh*nlocal).unsqueeze(dim=0) #nlcoal*nbatch
        votemask = votePass*votemask
    votemask = votemask.unsqueeze(dim=2)
    
    #import ipdb; ipdb.set_trace()
    if noweight:
        weight = votemask #nlcoal*batch*nclass
        weight = weight/weight.sum(dim=0)
        avgLogits = (weight*logits).sum(dim=0)
    else:
        if clscount:
            countweight = countN.unsqueeze(dim=1) #nlcoal*batch*nclass
        else:
            countweight = countN.sum(dim=1).unsqueeze(dim=1).unsqueeze(dim=2)
        weight = votemask*countweight #nlcoal*batch*nclass
        weight = weight/weight.sum(dim=0)
        avgLogits = (weight*logits).sum(dim=0)
    return avgLogits, votemask

class self_attention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(self_attention, self).__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_size = int(hidden_size / num_heads)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.mask = None
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, q, k, v, mask=None):
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.linear_o(context_layer)
        return context_layer
    
class MHA(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.self_attention_1 = self_attention(hidden_size, num_heads, dropout).cuda()
        self.self_attention_2 = self_attention(hidden_size, num_heads, dropout).cuda()
    
    def forward(self, fm_s, fm_t, mask=None):
        # 128 * 16 * 32 * 32 > 128 * 16 * 1024 
        fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
        # 128 * 16 * 32 * 32 > 128 * 16 * 1024 
        fm_t = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
        # self attention
        fm_s = self.self_attention_1(fm_s, fm_s, fm_s)
        fm_t = self.self_attention_2(fm_t, fm_t, fm_t)
        loss = (F.mse_loss(fm_s, fm_t))/math.sqrt(self.num_heads)
        return loss