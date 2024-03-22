
import torch
from torch import nn

class relup(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, alpha=0.014):
        super(relup, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        x = x.clamp(self.alpha)
        return x
    

class gemp(nn.Module):
    """ Reranking with maximum descriptors aggregation, can substitute regular GeM pooling
       https://github.com/ShihaoShao-GH/SuperGlobal/blob/main/modules/coarse_retrieval/rgem.py
    """
    def __init__(self, p=4.6, eps = 1e-8, emb_dim=None):
        super(gemp, self).__init__()
        self.p = p
        self.eps = eps
        self.avg_pool = nn.AdaptiveAvgPool2d((emb_dim, 1))
    def forward(self, x):
        x = x.clamp(self.eps).pow(self.p)
        x = self.avg_pool(x).pow(1. / (self.p) )
        return x

class rgem(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, pr=2.5, size = 5):
        super(rgem, self).__init__()
        self.pr = pr
        self.size = size
        self.lppool = nn.LPPool2d(self.pr, int(self.size), stride=1)
        self.pad = nn.ReflectionPad2d(int((self.size-1)//2.))
    def forward(self, x):
        nominater = (self.size**2) **(1./self.pr)
        x = 0.5*self.lppool(self.pad(x/nominater)) + 0.5*x
        return x


class gemp3D(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, p=4.6, eps = 1e-8, emb_dim=None):
        super(gemp3D, self).__init__()
        self.p = p
        self.eps = eps
        self.avg_pool = nn.AdaptiveAvgPool3d((emb_dim, 1, 1))
    def forward(self, x):
        x = x.clamp(self.eps).pow(self.p)
        x = self.avg_pool(x).pow(1. / (self.p) )
        return x

