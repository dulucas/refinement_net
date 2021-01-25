from config import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from seg_opr.seg_oprs import ConvBnRelu, RefineResidual
from base_model import resnet50_new
###########################################################

class Unet(nn.Module):
    def __init__(self, pretrained_model, norm_layer=nn.BatchNorm2d):
        super(Unet, self).__init__()
        BatchNorm2d = norm_layer

        self.encoder = resnet50_new(pretrained_model, norm_layer=BatchNorm2d, \
                                bn_eps=config.bn_eps,
                                bn_momentum=config.bn_momentum,
                                deep_stem=False, stem_width=64)

        self.down_scale = nn.MaxPool2d(2)
        self.up_scale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        ################################################################################

        self.mask_up_layer0 = RefineResidual(2048, 1024, 3, \
                                    norm_layer=BatchNorm2d, has_bias=True, has_relu=True)
        self.mask_up_layer1 = RefineResidual(1024, 512, 3, \
                                    norm_layer=BatchNorm2d, has_bias=True, has_relu=True)
        self.mask_up_layer2 = RefineResidual(512, 256, 3, \
                                    norm_layer=BatchNorm2d, has_bias=True, has_relu=True)

        self.mask_refine_layer0 = RefineResidual(256, 128, 3, \
                                    norm_layer=BatchNorm2d, \
                                    has_bias=True, has_relu=True)

        self.mask_refine_layer1 = RefineResidual(128, 64, 3, \
                                    norm_layer=BatchNorm2d, \
                                    has_bias=True, has_relu=True)

        self.mask_refine_layer2 = RefineResidual(64, 32, 3, \
                                    norm_layer=BatchNorm2d, \
                                    has_bias=True, has_relu=True)

        self.mask_output_layer = RefineResidual(32, 1, 3, \
                                    norm_layer=BatchNorm2d, \
                                    has_bias=False, has_relu=True)

       ################################################################################

    def forward(self, rgb):

        blocks = self.encoder(rgb)
        r4, r3, r2, r1 = blocks[-1], blocks[-2], blocks[-3], blocks[-4]

        ###########################
        x = self.up_scale(r4)
        x = self.mask_up_layer0(x)
        x = x + r3
        x = self.up_scale(x)
        x = self.mask_up_layer1(x)
        x = x + r2
        x = self.up_scale(x)
        x = self.mask_up_layer2(x)
        x = x + r1
        x = self.up_scale(x)
        x = self.mask_refine_layer0(x)
        x = self.mask_refine_layer1(x)
        x = self.up_scale(x)
        x = self.mask_refine_layer2(x)
        mask = self.mask_output_layer(x)

        ###########################
        return mask
