#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv

from nets.SATS1 import SATS1
from nets.SATS2 import SATS2
from nets.SATS3 import SATS3

from .sknet import SKConv



class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [256, 512, 1024], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        outputs = []
        for k, x in enumerate(inputs):
            x       = self.stems[k](x)
            cls_feat    = self.cls_convs[k](x)
            cls_output  = self.cls_preds[k](cls_feat)
            reg_feat    = self.reg_convs[k](x)
            reg_output  = self.reg_preds[k](reg_feat)
            obj_output  = self.obj_preds[k](reg_feat)
            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        self.SATS_P5 = SATS1()
        self.SATS_P4 = SATS2()
        self.SATS_P3 = SATS3()

        self.SATS_cnn_conv_P5  = BaseConv(int(in_channels[2] * width), int(in_channels[2] * width), 1, 1, act=act)
        self.BN_P5 = torch.nn.BatchNorm2d(int(in_channels[2] * width), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.P5conv = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)

        self.cnn_SATS_conv_P4 = BaseConv(int(in_channels[1] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.BN_P4_1 = torch.nn.BatchNorm2d(int(in_channels[1] * width), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.P4conv = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.SATS_cnn_conv_P4 = BaseConv(int(in_channels[1] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.BN_P4_2 = torch.nn.BatchNorm2d(int(in_channels[1] * width), eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)

        self.cnn_SATS_conv_P3 = BaseConv(int(in_channels[0] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.BN_P3_1 = torch.nn.BatchNorm2d(int(in_channels[0] * width), eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)
        self.P3conv = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.SATS_cnn_conv_P3 = BaseConv(int(in_channels[0] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.BN_P3_2 = torch.nn.BatchNorm2d(int(in_channels[0] * width), eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        self.SATS_cnn_conv_P3_2 = BaseConv(int(in_channels[0] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.BN_P3_3 = torch.nn.BatchNorm2d(int(in_channels[0] * width), eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)

        self.P3conv_2 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.cnn_SATS_conv_P3_2 = BaseConv(int(in_channels[0] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.BN_P3_4 = torch.nn.BatchNorm2d(int(in_channels[0] * width), eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)

        self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        self.SATS_cnn_conv_P4_2 = BaseConv(int(in_channels[1] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.BN_P4_3 = torch.nn.BatchNorm2d(int(in_channels[1] * width), eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)

        self.P4conv_2 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.cnn_SATS_conv_P4_2 = BaseConv(int(in_channels[1] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.BN_P4_4 = torch.nn.BatchNorm2d(int(in_channels[1] * width), eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)
        self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        self.SATS_cnn_conv_P5_2 = BaseConv(int(in_channels[2] * width), int(in_channels[2] * width), 1, 1, act=act)
        self.BN_P5_3 = torch.nn.BatchNorm2d(int(in_channels[2] * width), eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)

        self.P5conv_2 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.cnn_SATS_conv_P5_2 = BaseConv(int(in_channels[2] * width), int(in_channels[2] * width), 1, 1, act=act)
        self.BN_P5_4 = torch.nn.BatchNorm2d(int(in_channels[2] * width), eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)

        self.P3_SKConv = SKConv(int(in_channels[0] * width))
        self.P4_SKConv = SKConv(int(in_channels[1] * width))
        self.P5_SKConv = SKConv(int(in_channels[2] * width))

    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        feat3_SATS = self.SATS_P5(feat3)
        feat3_SATS = feat3_SATS.permute(0, 2, 1)

        feat2_SATS = self.SATS_P4(feat2)
        feat2_SATS = feat2_SATS.permute(0, 2, 1)

        feat1_SATS = self.SATS_P3(feat1)
        feat1_SATS = feat1_SATS.permute(0, 2, 1)

        P5_SATS = feat3_SATS.reshape([-1, 512, 20, 20])
        P5_cat = self.SATS_cnn_conv_P5(P5_SATS)
        P5_cat = self.BN_P5(P5_cat)

        P5_cat = torch.cat([feat3, P5_cat], 1)
        P5_cat = self.P5conv(P5_cat)
        P5          = self.lateral_conv0(P5_cat)
        P5_upsample = self.upsample(P5)

        P5_upsample = self.cnn_SATS_conv_P4(P5_upsample)
        P5_upsample = self.BN_P4_1(P5_upsample)

        P4_SATS = feat2_SATS.reshape([-1, 256, 40, 40])
        P4_SATS = torch.cat([P4_SATS, P5_upsample], 1)
        P4_SATS = self.P4conv(P4_SATS)

        P4_cat = self.SATS_cnn_conv_P4(P4_SATS)
        P4_cat = self.BN_P4_2(P4_cat)
        P5_upsample_2 = torch.cat([P4_cat, feat2], 1)
        P5_upsample_2 = self.C3_p4(P5_upsample_2)
        P4          = self.reduce_conv1(P5_upsample_2)
        P4_upsample = self.upsample(P4)
        P4_upsample = self.cnn_SATS_conv_P3(P4_upsample)
        P4_upsample = self.BN_P3_1(P4_upsample)
        P3_SATS = feat1_SATS.reshape([-1, 128, 80, 80])
        P3_SATS = torch.cat([P3_SATS, P4_upsample], 1)
        P3_SATS = self.P3conv(P3_SATS)
        P3_cat = self.SATS_cnn_conv_P3(P3_SATS)
        P3_cat = self.BN_P3_2(P3_cat)
        P4_upsample_2 = torch.cat([P3_cat, feat1], 1)
        P3_out      = self.C3_p3(P4_upsample_2)
        P3_downsample = self.SATS_cnn_conv_P3_2(P3_out)
        P3_downsample = self.BN_P3_3(P3_downsample)
        P3_SATS_2 = torch.cat([P3_SATS, P3_downsample], 1)
        P3_SATS_out = self.P3conv_2(P3_SATS_2)
        P3_downsample_2 = self.cnn_SATS_conv_P3_2(P3_SATS_out)
        P3_downsample_2 = self.BN_P3_4(P3_downsample_2)
        P3_downsample_2   = self.bu_conv2(P3_downsample_2)
        P3_downsample_2   = torch.cat([P3_downsample_2, P4], 1)
        P4_out          = self.C3_n3(P3_downsample_2)
        P4_downsample = self.SATS_cnn_conv_P4_2(P4_out)
        P4_downsample = self.BN_P4_3(P4_downsample)
        P4_SATS_2 = torch.cat([P4_SATS, P4_downsample], 1)
        P4_SATS_out = self.P4conv_2(P4_SATS_2)
        P4_downsample_2 = self.cnn_SATS_conv_P4_2(P4_SATS_out)
        P4_downsample_2 = self.BN_P4_4(P4_downsample_2)
        P4_downsample_2   = self.bu_conv1(P4_downsample_2)
        P4_downsample_2   = torch.cat([P4_downsample_2, P5], 1)
        P5_out          = self.C3_n4(P4_downsample_2)
        P5_downsample = self.SATS_cnn_conv_P5_2(P5_out)
        P5_downsample = self.BN_P5_3(P5_downsample)
        P5_SATS_2 = torch.cat([P5_SATS, P5_downsample], 1)
        P5_SATS_out = self.P5conv_2(P5_SATS_2)

        P3_out = self.P3_SKConv(P3_out, P3_SATS_out)
        P4_out = self.P4_SKConv(P4_out, P4_SATS_out)
        P5_out = self.P5_SKConv(P5_out, P5_SATS_out)


        return (P3_out, P4_out, P5_out)

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False 

        self.backbone   = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs    = self.backbone.forward(x)
        outputs     = self.head.forward(fpn_outs)
        return outputs
