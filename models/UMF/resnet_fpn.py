import torch.nn as nn
import torch.nn.functional as F
import torch
import timm

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResNetFPN(nn.Module):
    """
    ResNet+FPN, output resolution are 1/16 and 1/4.
    Each block has 2 layers.
    """

    def __init__(self, slug, input_dim=3, fpn=True):
        super().__init__()
        # Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if slug == "resnet18":
            block_dims = [64, 128, 256, 512]
        
        elif slug == "resnet50":
            block_dims = [256, 512, 1024, 2048]

        slug = "se" + slug
        self.fpn = fpn
        self.resnet = timm.create_model(slug, pretrained=True, in_chans=input_dim)

        if input_dim != 3:
            print("Changing input channels from 3 to {}".format(input_dim))
            self.resnet.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if fpn:
            self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
            self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
            self.layer3_outconv2 = nn.Sequential(
                conv3x3(block_dims[3], block_dims[3]),
                nn.BatchNorm2d(block_dims[3]),
                nn.LeakyReLU(),
                conv3x3(block_dims[3], block_dims[2]),
            )

            self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
            self.layer2_outconv2 = nn.Sequential(
                conv3x3(block_dims[2], block_dims[2]),
                nn.BatchNorm2d(block_dims[2]),
                nn.LeakyReLU(),
                conv3x3(block_dims[2], block_dims[1]),
            )

    def load_checkpoint(self, ckpt_path):
        print("Loading checkpoint from {}".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.load_state_dict(checkpoint, strict=False)
        print("Loaded checkpoint from {}".format(ckpt_path))

    
    def freeze_backbone(self):
        for param in self.resnet.parameters():
            param.requires_grad = False


    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        y = {}
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = F.relu(x0)
        #x0 = self.resnet.maxpool(x0)

        x1 = self.resnet.layer1(x0)  # 1/2
        x2 = self.resnet.layer2(x1)  # 1/4
        x3 = self.resnet.layer3(x2)  # 1/8
        x4 = self.resnet.layer4(x3)  # 1/16

        if not self.fpn:
            y["coarse"] = x4
            return y
        
        # FPN
        x4_out = self.layer4_outconv(x4)

        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out+x4_out_2x)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        #y["fine_1"] = x3_out
        y["fine_2"] = x2_out
        y["coarse"] = x4
        return y
