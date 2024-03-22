import torch
from torch import nn
import torch.nn.functional as F
from .pooling import rgem, gemp, relup
from .swin_transformer import TransformerEncoder
from how import layers


class LocalFeatureTransformer(nn.Module):
    def __init__(self, input_dim, out_dim, num_heads=8, depth=2, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.transformer_encoder = TransformerEncoder(input_dim, depth=depth, drop_path=0.0,
            num_heads=num_heads, window_size=(7, 7))

        self.pool = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=3, stride=2, padding=0),
            relup(),
            rgem(),
            gemp(emb_dim=1),
            nn.Flatten()
        )
        self.reduction_layer = layers.dim_reduction.ConvDimReduction(dim=out_dim, input_dim=input_dim)

    def forward(self, x, dim_reduction=False, normalize=False):
        embed = x

        out, attn_maps = self.transformer_encoder(embed)
        embeddings = self.pool(out)

        if normalize:
            out = F.normalize(out, p=2, dim=-1)
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        if dim_reduction:
            with torch.no_grad():
                out = self.reduction_layer(out)

        return embeddings, out.detach(), attn_maps


    def __repr__(self):
        s = str(self.__class__.__name__)
        for k in ["input_dim","out_dim"]:
            s += "\n  {:s}: {:d}".format(k, getattr(self,k))
        return s