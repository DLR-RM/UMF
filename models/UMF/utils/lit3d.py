# Copyright (C) 2021-2022 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
from torch import nn
import math
from how import layers
import torch.nn.functional as F


class LocalfeatureIntegrationTransformer3D(nn.Module):
    """Map a set of local features to a fixed number of SuperFeatures """

    def __init__(self, T, N, input_dim, feat_dim, out_dim):
        """
        T: number of iterations
        N: number of SuperFeatures
        input_dim: dimension of input local features
        dim: dimension of SuperFeatures
        """
        super().__init__()
        self.T = T
        self.N = N
        self.input_dim = input_dim
        dim = feat_dim
        hidden_dim = input_dim*2

        # learnable initialization
        self.templates_init = nn.Parameter(torch.randn(1, self.N, dim))
        self.project = nn.Sequential(
            nn.Conv3d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

        # qkv
        self.project_q = nn.Linear(dim, dim, bias=False)
        self.project_k = nn.Linear(hidden_dim, dim, bias=False)
        self.project_v = nn.Linear(hidden_dim, dim, bias=False)
        # layer norms
        self.norm_inputs = nn.LayerNorm(hidden_dim)
        self.norm_templates = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = dim ** -0.5
        # mlp
        self.norm_mlp = nn.LayerNorm(dim)
        mlp_dim = dim*2
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim) )
        self.reduction_layer = layers.dim_reduction.LinearDimReduction(dim=out_dim, input_dim=dim)


    def forward(self, x, dim_reduction=False):
        """
        input:
            x has shape BxCxXxYxZ
        output:
            template (output SuperFeatures): tensor of shape BxCxNx1
            attn (attention over local features at the last iteration): tensor of shape BxNxXxYxZ
        """
        x = self.project(x)
        B, C, X, Y, Z = x.size()

        x = x.reshape(B, C, X * Y * Z).permute(0, 2, 1)

        x = self.norm_inputs(x)
        k = self.project_k(x)
        v = self.project_v(x)

        # template initialization
        templates = torch.repeat_interleave(self.templates_init, B, dim=0)
        attn = None

        # main iteration loop
        for _ in range(self.T):
            templates_prev = templates

            # q projection
            templates = self.norm_templates(templates)
            q = self.project_q(templates)

            q = q * self.scale  # Normalization.
            
            attn_logits = torch.einsum('bnd,bld->bln', q, k)
            attn = self.softmax(attn_logits)
            attn = attn + 1e-8  # to avoid zero when with the L1 norm below
            attn = attn / attn.sum(dim=-2, keepdim=True)

            # update template
            templates = templates_prev + torch.einsum('bld,bln->bnd', v, attn)

            # mlp
            templates = templates + self.mlp(self.norm_mlp(templates))

        # reshape templates to BxDxNx1
        templates = templates.permute(0,2,1)[:,:,:]
        attn = attn.permute(0,2,1).view(B,self.N,X,Y,Z)

        if dim_reduction:
            with torch.no_grad():
                templates = self.reduction_layer(templates)
        
        return templates, attn
    
    def __repr__(self):
        s = str(self.__class__.__name__)
        for k in ["T","N","input_dim","dim"]:
            s += "\n  {:s}: {:d}".format(k, getattr(self,k))
        return s