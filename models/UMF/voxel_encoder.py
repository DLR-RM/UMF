from functools import partial

import torch
import torch.nn as nn
import numpy as np

from models.UMF.utils.spconv_utils import replace_feature, spconv
from .spconv_backbone import post_act_block
import torch.nn.functional as F
import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True

from functools import partial

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key, algo=spconv.ConvAlgo.Native
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key, algo=spconv.ConvAlgo.Native
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out

class GeM3D(nn.Module):
    def __init__(self, input_dim, p=3, eps=1e-6):
        super(GeM3D, self).__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: spconv.SparseConvTensor):
        # Convert SparseConvTensor to dense tensor (B, C, Z, Y, X) format
        dense_x = x.dense()

        # Clamp, raise to power p, and average pool over spatial dimensions
        pooled = F.avg_pool3d(dense_x.clamp(min=self.eps).pow(self.p),
                              kernel_size=dense_x.size()[2:])

        # Squeeze spatial dimensions and raise to power 1/p
        output = pooled.squeeze(-1).squeeze(-1).squeeze(-1).pow(1./self.p)

        return output


class Voxel_MAE(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size, point_cloud_range, fpn=False, **kwargs):
        print("Voxel_MAE init", input_channels, grid_size, voxel_size, point_cloud_range, fpn,  kwargs)
        super().__init__()
        self.sparse_shape = np.array(grid_size[::-1])
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.fpn = fpn
        self.frozen = False
        self.freeze_fpn = fpn
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_channels = 1

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1', algo=spconv.ConvAlgo.Native),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            # 200, 200, 200
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # 100, 100, 100
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(1, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0

        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2', algo=spconv.ConvAlgo.Native),
            norm_fn(128),
            nn.ReLU(),
        )


        if self.fpn:
            self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
            self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
            self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

            self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
            self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
            self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

            self.conv3_up = spconv.SparseSequential(
                block(64, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
            )
            
        self.num_point_features = 32


    def freeze_backbone(self):
        self.frozen = True
        for param in self.conv_input.parameters():
            param.requires_grad = False
        
        for param in self.conv1.parameters():
            param.requires_grad = False
        
        for param in self.conv2.parameters():
            param.requires_grad = False

        for param in self.conv3.parameters():
            param.requires_grad = False

        for param in self.conv4.parameters():
            param.requires_grad = False

        for param in self.conv_out.parameters():
            param.requires_grad = False

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def load_checkpoint(self, ckpt_path):
        print("Loading checkpoint from {}".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.load_state_dict(checkpoint, strict=False)
        print("Loaded checkpoint from {}".format(ckpt_path))

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'].to(self.device), batch_dict['coordinates'].to(self.device)
        
        batch_size = batch_dict['batch_size']


        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        out = self.conv_out(x_conv4)
        batch_dict["coarse"] = out.dense() 

        if not self.fpn:
            return batch_dict
        
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.conv3_up)
        
        # 50, 50, 50
        #batch_dict["fine_1"] =  x_up4.dense()
        batch_dict["fine_2"] =  x_up3.dense()


        return batch_dict