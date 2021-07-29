
import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, FeatureHeatmapFusingBlock
from .modules.architecture import FeedbackHourGlass
from .modules.architecture import StackedHourGlass
from .srfbn_hg_arch import  FeedbackBlockHeatmapAttention, merge_heatmap_5, FeedbackBlockCustom
import numpy as np
import matplotlib.pyplot as plt


class MLG(nn.Module):
    def __init__(self, opt, device):
        super(MLG,self).__init__()
        in_channels = opt['in_channels']
        out_channels = opt['out_channels']
        num_groups = opt['num_groups']
        hg_num_feature = opt['hg_num_feature']
        hg_num_keypoints = opt['hg_num_keypoints']
        act_type = 'prelu'
        norm_type = None

        self.num_steps = opt['num_steps']
        num_features = opt['num_features']
        self.upscale_factor = opt['scale']
        self.detach_attention = opt['detach_attention']
        if self.detach_attention:
            print('Detach attention!')
        else:
            print('Not detach attention!')

        if self.upscale_factor == 8:

            stride = 4
            padding = 2
            kernel_size = 8
        else:
            raise NotImplementedError("Upscale factor %d not implemented!" % self.upscale_factor)

        # LR feature extraction block
        self.conv_in = ConvBlock(
            in_channels,
            4 * num_features,
            kernel_size=3,
            act_type=act_type,
            norm_type=norm_type)
        self.feat_in = nn.PixelShuffle(2)

        # basic block
        self.first_block = FeedbackBlockCustom(num_features, num_groups, self.upscale_factor,
                                   act_type, norm_type, num_features)



        self.atten_block = FeedbackBlockHeatmapAttention(num_features, num_groups, self.upscale_factor, act_type, norm_type, 5, opt['num_fusion_block'], device=device)
        self.atten_block.should_reset =False


        self.out = DeconvBlock(
            num_features,
            num_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_type='prelu',
            norm_type=norm_type)

        self.conv_out = ConvBlock(
            num_features,
            out_channels,
            kernel_size=3,
            act_type=None,
            norm_type=norm_type)

        self.HG = FeedbackHourGlass(hg_num_feature, hg_num_keypoints)
    def forward(self, x):
        inter_res = nn.functional.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode='bilinear',
            align_corners=False)
        x = self.conv_in(x)
        x = self.feat_in(x)

        sr_outs = []
        heatmap_outs = []
        merge_heatmap=[]
        hg_last_hidden = None
        # initalize heatmap and FB feature with first coarse block
        for step in range(self.num_steps):
            if step == 0:
                FB_out_first = self.first_block(x)
                h = torch.add(inter_res, self.conv_out(self.out(FB_out_first)))

                heatmap, hg_last_hidden = self.HG(h, hg_last_hidden)

                tmp, one_heatmap = merge_heatmap_5(heatmap,self.detach_attention)

                self.atten_block.last_hidden = FB_out_first

                assert  self.atten_block.should_reset == False
            else:
                tmp, one_heatmap = merge_heatmap_5(heatmap, self.detach_attention)
                FB_out = self.atten_block(x,tmp)

                h = torch.add(inter_res, self.conv_out(self.out(FB_out)))

                heatmap, hg_last_hidden = self.HG(h, hg_last_hidden)

            sr_outs.append(h)
            heatmap_outs.append(heatmap)
            merge_heatmap.append(one_heatmap)

        return sr_outs, heatmap_outs,merge_heatmap
