import torch
EPS = 1e-5

class kl_loss(torch.nn.Module):
    '''
    summary : kullback-leibler divergence loss      
    '''
    def __init__(self, T=3, singlelabel=False):
        super().__init__()
        self.T = T
        self.singlelabel = singlelabel
        self.criterion= torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, input, target):
        if self.singlelabel:
            soft_in = torch.log_softmax(input/self.T, dim=1)
            soft_target = torch.softmax(target/self.T, dim=1).float()
            loss = self.criterion(soft_in, soft_target)
        else:
            soft_in = torch.sigmoid(input/self.T)
            soft_target = torch.sigmoid(target/self.T).float()
            loss = self.criterion(soft_in.log(), soft_target)
        return self.T*self.T*loss

class KLDivLoss(torch.nn.KLDivLoss):
    def __init__(self, reduction='none', T=2, singlelabel=False):
        super().__init__(reduction=reduction)
        self.T = T
        self.singlelabel = singlelabel

    def forward(self, input, target):
        if self.singlelabel:
            preds_ = torch.log_softmax(input/self.T, dim=1)
            targets_ = torch.softmax(target/self.T, dim=1)
        
        else:
            preds_ = torch.sigmoid(input/self.T)
            targets_ = torch.sigmoid(target/self.T).float()
            
        return super(KLDivLoss, self).forward(preds_, targets_)

def weight_psedolabel(logits, countN, noweight=False, clscount=False, votethresh=0, singlabel=True):#nlcoal*batch*nclass
    #softLogits = torch.nn.Softmax(dim=2)(logits)
    nlocal, nbatch, ncls = logits.shape
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