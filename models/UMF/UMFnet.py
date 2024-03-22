import sys
import torch
import torch.nn as nn
import numpy as np
from how import layers
import yaml
import dotsi
from models.UMF.resnet_fpn import ResNetFPN
from models.UMF.utils.local_transformer import LocalFeatureTransformer
from models.UMF.utils.local_transformer3D import LocalFeatureTransformer3D
from models.UMF.utils.lit import LocalfeatureIntegrationTransformer
from models.UMF.utils.lit3d import LocalfeatureIntegrationTransformer3D
from models.UMF.voxel_encoder import Voxel_MAE
from models.UMF.utils.multimodal_fusion import FusionEncoder


def load_config(config_file):
    with open(config_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


class UMFnet(torch.nn.Module):
    def __init__(self, cfg, grid_size=None, voxel_size=None, final_block: str = None):
        super().__init__()
        print(" final_block: ", final_block)
        point_cloud_range = cfg.model.point_cloud.range

        self.cloud_fe_size = cfg.model.point_cloud.out_dim
        self.image_fe_size = cfg.model.visual.out_dim
        pc_input_dim = cfg.model.point_cloud.input_dim
        img_input_dim = cfg.model.visual.input_dim
        cfg = dotsi.Dict(cfg)
        self.cfg = cfg
        self.fusion_only = cfg.model.mode== "fusion"

        grid_size = cfg.model.point_cloud.grid_size
        self.cloud_fe = Voxel_MAE(pc_input_dim,  grid_size, voxel_size, point_cloud_range, 
                                  fpn=not self.fusion_only)

        vis_model_name = cfg.model.visual.architecture
        self.image_fe = ResNetFPN(vis_model_name, input_dim=img_input_dim,  
                                fpn=not self.fusion_only)

        visual_dim = 2048 if vis_model_name == "resnet50" else 512
        lidar_dim = 128

        d_attn = cfg.model.fusion.d_attn
        num_heads = cfg.model.fusion.num_heads
        self.fusion_encoder = FusionEncoder(visual_dim, lidar_dim, d_attn, num_heads)
        self.output_dim = cfg.model.fusion.d_embedding
        self.final_block = final_block


        self.fused_dim = d_attn
        if self.final_block is None:
            self.final_net = None
        elif self.final_block == 'fc':
            self.final_net = nn.Linear(self.fused_dim, self.output_dim)

        elif self.final_block == 'mlp':
            temp_channels = self.output_dim
            self.final_net = nn.Sequential(nn.Linear(self.fused_dim, temp_channels, bias=False),
                                           nn.BatchNorm1d(temp_channels, affine=True),
                                           nn.ReLU(inplace=True), nn.Linear(temp_channels, self.output_dim))
        else:
            raise NotImplementedError('Unsupported final block: {}'.format(self.final_block))

        # FPN dim
        visual_dim = 512 if vis_model_name == "resnet50" else 128
        lidar_dim = 32

        if not self.fusion_only:
            if cfg.model.mode == "superfeatures":
                lit_cfg_im = cfg.model.visual.local_superfeatures
                lit_cfg_pc = cfg.model.point_cloud.local_superfeatures

                self.runtime = cfg.model.visual.local_superfeatures.runtime

                self.lit = LocalfeatureIntegrationTransformer(lit_cfg_im.T, lit_cfg_im.N, 
                                                              visual_dim, feat_dim=lit_cfg_im.dim,
                                                              out_dim=lit_cfg_im.out_dim)
                self.lit3d = LocalfeatureIntegrationTransformer3D(lit_cfg_pc.T, lit_cfg_pc.N, 
                                                                    lidar_dim, 
                                                                    feat_dim=lit_cfg_pc.dim,
                                                                    out_dim=lit_cfg_pc.out_dim)
                if cfg.model.smoothing:
                    self.smoothing = layers.smoothing.Smoothing()
                self.attention = layers.attention.L2Attention()
                self.attention3d = layers.attention.L2Attention()

            elif cfg.model.mode == "ransac":
                dim_local_feat_im = cfg.model.visual.local_ransac.dim
                dim_local_feat_pc = cfg.model.point_cloud.local_ransac.dim
                self.lt = LocalFeatureTransformer(input_dim=visual_dim, out_dim=dim_local_feat_im)
                self.lt3d = LocalFeatureTransformer3D(input_dim=lidar_dim, out_dim=dim_local_feat_pc)
            else:
                raise NotImplementedError('Unsupported mode: {}'.format(cfg.model.mode))
        
        if cfg.model.visual.pretrained:
            print("load visual pretrained model: ", cfg.model.visual.pretrained)
            self.image_fe.load_checkpoint(cfg.model.visual.pretrained,)


        if cfg.model.point_cloud.pretrained:
            print("load point cloud pretrained model: ", cfg.model.point_cloud.pretrained)
            self.cloud_fe.load_checkpoint(cfg.model.point_cloud.pretrained)

        if cfg.model.pretrained:
            path_ckpt = cfg.model.pretrained
            pretrained_state_dict = torch.load(path_ckpt, map_location='cpu')
            dim_reduction_prefix = '.reduction_layer.' 
            filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if dim_reduction_prefix not in k}
            
            self.load_state_dict(filtered_state_dict, strict=False)
            print("--> UMF - load pretrained model: ", path_ckpt)

        #self.image_fe.freeze_backbone()
        #self.cloud_fe.freeze_backbone()
        #self.freeze_fusion_branch()

    def freeze_fusion_branch(self):
        for param in self.fusion_encoder.parameters():
            param.requires_grad = False


    def get_superfeatures(self, x, dim_reduction=False):
        """
        return a list of tuple (features, attentionmpas) where each is a list containing requested scales
        features is a tensor BxDxNx1
        attentionmaps is a tensor BxNxHxW
        """
        feats = []
        attns = []
        strengths = []

        o, attn = self.lit(x, dim_reduction=dim_reduction)
        strength = self.attention(o)
        if self.cfg.model.smoothing:
            o = self.smoothing(o)

        feats = o.permute(0, 2, 1)
        attns = attn
        if strength.dim() != 3:
            strength = strength.unsqueeze(0)

        strengths = strength
        return feats, attns, strengths



    def get_local_features(self, x,dim_reduction=False, normalize=False):
        embed, feats, attn = self.lt(x, dim_reduction=dim_reduction, normalize=normalize)
        return embed, feats, attn
        

    def get_superfeatures_points(self, x, dim_reduction=False):
        """
        return a list of tuple (features, attentionmpas) where each is a list containing requested scales
        features is a tensor BxDxNx1
        attentionmaps is a tensor BxNxHxW
        """
        feats = []
        attns = []
        strengths = []

        o, attn = self.lit3d(x, dim_reduction=dim_reduction)
        strength = self.attention3d(o)
        if self.cfg.model.smoothing:
            o = self.smoothing(o)
        feats = o.permute(0, 2, 1)
        attns = attn
        if strength.dim() != 3:
            strength = strength.unsqueeze(0)
        strengths = strength
        return feats, attns, strengths
    

    def get_local_features_3d(self, x, dim_reduction=False, normalize=False):
        embed, feats, attn = self.lt3d(x, dim_reduction=dim_reduction, normalize=normalize)
        return embed, feats, attn
    

    def forward(self, batch, dim_reduction=False, normalize=False):
        y = {}
        y_img = self.image_fe(batch["images"])

        # cloud features
        batch_dict = {
            "voxel_features": batch["voxel_features"],
            "coordinates": batch["coordinates"],
            "batch_size": batch["images"].shape[0],
        }
        y_cloud = self.cloud_fe(batch_dict)

        if not self.fusion_only:
            fine_img = y_img['fine_2']
            fine_cloud = y_cloud['fine_2']

        im_feat = y_img['coarse'] 
        pc_feat = y_cloud['coarse']
        out = self.fusion_encoder(im_feat, pc_feat)

        if self.final_block is not None:
            out = self.final_net(out)
        y['embedding'] = out
        
        if self.cfg.model.mode == "superfeatures":
            img_super_feat, img_attns, img_strengths = self.get_superfeatures(fine_img, dim_reduction=dim_reduction)

            y['img_super_feat'] = img_super_feat
            y["img_attns"] = img_attns
            y["img_strengths"] = img_strengths

            pc_super_feat, pc_attns, pc_strengths = self.get_superfeatures_points(fine_cloud, dim_reduction=dim_reduction)
            y['pc_super_feat'] = pc_super_feat
            y["pc_attns"] = pc_attns
            y["pc_strengths"] = pc_strengths


        if self.cfg.model.mode == "ransac":
            # Visual local
            img_embed, img_local_feat, img_attn = self.get_local_features(fine_img,
                                                        dim_reduction=dim_reduction, normalize=normalize)
            y['img_local_feat'] = img_local_feat
            y['image_embedding'] = img_embed
            y['img_fine_feat'] = img_local_feat
            y["img_attns"] = img_attn

            pc_embed, pc_local_feat, pc_attns = self.get_local_features_3d(fine_cloud,
                                                        dim_reduction=dim_reduction, normalize=normalize)
            y['pc_local_feat'] = pc_local_feat
            y['pc_fine_feat'] = pc_local_feat
            y["pc_attns"] = pc_attns
            y['cloud_embedding'] = pc_embed

        return y

    def print_info(self):
        print('Model class: UMFnet')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))

        n_params = sum([param.nelement() for param in self.image_fe.parameters()])
        print('Image feature extractor parameters: {}'.format(n_params))

        n_params = sum([param.nelement() for param in self.cloud_fe.parameters()])
        print('Cloud feature extractor parameters: {}'.format(n_params))

        # Fusion branch
        if hasattr(self, 'fusion_encoder'):
            n_params = sum([param.nelement() for param in self.fusion_encoder.parameters()])
            print('Fusion model parameters: {}'.format(n_params))


        print('Final block: {}'.format(self.final_block))
        print('Dimensionality of cloud features: {}'.format(self.cloud_fe_size))
        print('Dimensionality of image features: {}'.format(self.image_fe_size))
        print('Dimensionality of final descriptor: {}'.format(self.output_dim))



