import torch
import torch.nn as nn
from models.UMF.utils.pooling import rgem, relup, gemp, gemp3D


class FusionEncoder(nn.Module):
    """Image and Voxel Fusion Encoder for multimodal fusion bnanch"""

    def __init__(self, visual_dim, lidar_dim, d_attn, num_heads):
        super().__init__()

        emb_dim = 64
        self.visual_pos_enc = nn.Parameter(torch.randn(emb_dim, d_attn))
        self.lidar_pos_enc = nn.Parameter(torch.randn(emb_dim, d_attn))

        # Encoder for visual modality
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(visual_dim, d_attn, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_attn),
            relup(),
            rgem(),
            gemp(emb_dim=emb_dim),
        )

        # Encoder for lidar modality
        self.lidar_encoder = nn.Sequential(
            nn.Conv3d(lidar_dim, d_attn, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(d_attn),
            relup(),
            gemp3D(emb_dim=emb_dim),
        )

        self.cross_attn1 = nn.MultiheadAttention(d_attn, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_attn)
        self.layer_norm11 = nn.LayerNorm(d_attn)

        self.self_attn1 = nn.MultiheadAttention(d_attn, num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(d_attn)

        self.cross_attn2 = nn.MultiheadAttention(d_attn, num_heads, batch_first=True)
        self.layer_norm3 = nn.LayerNorm(d_attn)

        self.self_attn2 = nn.MultiheadAttention(d_attn, num_heads, batch_first=True)
        self.layer_norm4 = nn.LayerNorm(d_attn)

        self.cross_attn3 = nn.MultiheadAttention(d_attn, num_heads, batch_first=True)
        self.layer_norm5 = nn.LayerNorm(d_attn)

        self.pool = nn.AdaptiveAvgPool1d(1)

        
    def forward(self, visual_input, lidar_input):
        im_feat = self.visual_encoder(visual_input)
        pc_feat = self.lidar_encoder(lidar_input)

        im_feat = im_feat.squeeze(-1).permute(0, 2, 1)
        pc_feat = pc_feat.squeeze(-1).squeeze(-1).permute(0, 2, 1)

        im_feat = im_feat + self.visual_pos_enc
        pc_feat = pc_feat + self.lidar_pos_enc

        pc_feat = self.layer_norm1(pc_feat)
        im_feat = self.layer_norm11(im_feat)

        x, _ = self.cross_attn1(pc_feat, im_feat, im_feat)
        fused_feat = x + pc_feat  # skip connection
        
        fused_feat = self.layer_norm2(fused_feat)
        x, _ = self.self_attn1(fused_feat, fused_feat, fused_feat)
        fused_feat = x + fused_feat  # skip connection
        pc_skip = fused_feat
        
        fused_feat = self.layer_norm3(fused_feat)
        x, _ = self.cross_attn2(fused_feat, im_feat, im_feat)
        fused_feat = x + fused_feat  # skip connection

        fused_feat = self.layer_norm4(fused_feat)
        x, _ = self.self_attn2(fused_feat, fused_feat, fused_feat)
        fused_feat = x + fused_feat  # skip connection

        fused_feat = self.layer_norm5(fused_feat)
        x, _ = self.cross_attn3(fused_feat, pc_skip, pc_skip)
        x = x + fused_feat  # skip connection

        embedding = self.pool(x.permute(0, 2, 1)).squeeze(-1)  # B x F
        return embedding
