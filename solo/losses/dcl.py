import numpy as np
import torch
from pdb import set_trace as st
from torch import nn
import torch.nn.functional as F


SMALL_NUM = np.log(1e-45)

def acos_safe(x, eps=1e-4):
    slope = np.arccos(1-eps) / eps
    # TODO: stop doing this allocation once sparse gradients with NaNs (like in
    # th.where) are handled differently.
    buf = torch.empty_like(x).float()
    good = abs(x) <= 1-eps
    bad = ~good
    sign = torch.sign(x[bad])
    buf[good] = torch.acos(x[good])
    buf[bad] = torch.acos(sign * (1 - eps)) - slope*sign*(abs(x[bad]) - 1 + eps)
    return buf

class Z_loss(nn.Module):
    def __init__(self,):
        super().__init__()
        pass
        
    def forward(self,z):
        z_list = z.chunk(2,dim=0)
        z_sim = F.cosine_similarity(z_list[0],z_list[1],dim=1).mean()
        return -z_sim
    
class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)
    
class PoseLoss_cos(object):
    def __init__(self, temperature=0.1, weight_fn=None):
        super(PoseLoss, self).__init__()    
    
    def dist(self, z1, z2):
        return acos_safe(torch.bmm(z1.unsqueeze(1), z2.unsqueeze(-1)).squeeze())
    
    def __call__(self, za, zl, zr):
        f13 = self.dist(zl, zr) #torch.acos(torch.bmm(zl.unsqueeze(1), zr.unsqueeze(-1)).squeeze())
        f12_1 = self.dist(za, zl)
        f23_1 = self.dist(za, zr)
        
        return ((f12_1+f23_1-f13)**2).mean()

class PoseLoss(object):
    def __init__(self, temperature=0.1, weight_fn=None):
        super(PoseLoss, self).__init__()
    
    def dist(self, u1, u2):
        tmp = torch.matmul(u1, torch.transpose(u2, 0, 1))
        return torch.diagonal(tmp)
    
    def __call__(self, ep, e1, e2):
        v1 = ep - e1
        v2 = e2 - ep
        
        d1 = torch.bmm(v1.unsqueeze(-2), ep.unsqueeze(-1)).squeeze()
        d2 = torch.bmm(v2.unsqueeze(-2), ep.unsqueeze(-1)).squeeze()
        
        u1 = v1 - d1.unsqueeze(1) * ep
        u2 = v2 - d2.unsqueeze(1) * ep
        res = self.dist(u1, u2)
        return res.mean()
    
class PoseLoss_dual(object):
    def __init__(self, temperature=0.1, weight_fn=None):
        super(PoseLoss, self).__init__()    
    
    def dist(self, z1, z2):
        return torch.acos(torch.bmm(z1.unsqueeze(1), z2.unsqueeze(-1)).squeeze())
    
    def __call__(self, z1, z2, zl, zr):
        f13 = self.dist(zl, zr) #torch.acos(torch.bmm(zl.unsqueeze(1), zr.unsqueeze(-1)).squeeze())
        
        f12_1 = self.dist(z1, zl)
        f23_1 = self.dist(z1, zr)
        f12_2 = self.dist(z2, zl)
        f23_2 = self.dist(z2, zr)
        
        return ((f12_1+f23_1-f13)**2+(f12_2+f23_2-f13)**2).mean()
        
class DCL(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()


class DCLW(DCL):
    """
    Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, sigma=0.5, temperature=0.1):
        weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        super(DCLW, self).__init__(weight_fn=weight_fn, temperature=temperature)
