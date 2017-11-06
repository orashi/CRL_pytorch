import numpy as np
import torch
import os
import sys
import functools
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F
import torchvision.models as M


class Conv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)

    def forward(self, x):
        return F.leaky_relu(self.conv.forward(x), negative_slope=0.1, inplace=True)


class TConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1):
        super(TConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)

    def forward(self, x):
        return F.leaky_relu(self.conv.forward(x), negative_slope=0.1, inplace=True)



class DResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
        """
        super(DResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return residual + bottleneck


class Tunnel(nn.Module):
    def __init__(self, len=1, *args):
        super(Tunnel, self).__init__()

        tunnel = [DResNeXtBottleneck(*args) for _ in range(len)]
        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
        return self.tunnel(x)


class DilateTunnel(nn.Module):
    def __init__(self, depth=4):
        super(DilateTunnel, self).__init__()

        tunnel = [ResNeXtBottleneck(dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(dilate=8) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(dilate=1) for _ in range(14)]

        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
        return self.tunnel(x)


class def_netG(nn.Module):
    def __init__(self, ngf=64):
        super(def_netG, self).__init__()

        ################ down
        self.conv1 = Conv(3, ngf, kernel_size=7, stride=2, padding=1)
        self.conv2 = Conv(ngf, ngf * 2, kernel_size=5, stride=2, padding=1)

        self.corr = nn.Sequential(nn.Conv2d(ngf * 4, 81, kernel_size=1, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv_rdi = nn.Sequential(nn.Conv2d(ngf * 2, ngf, kernel_size=1, stride=1, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv3 = Conv(145, ngf * 4, kernel_size=5, stride=2, padding=1)
        self.conv3_1 = Conv(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = Conv(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.conv5 = Conv(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = Conv(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.conv6 = Conv(ngf * 8, ngf * 16, kernel_size=3, stride=2, padding=1)
        self.conv6_1 = Conv(ngf * 16, ngf * 16, kernel_size=3, stride=1, padding=1)

        ################ extract
        self.pr64 = Conv(ngf * 16, 1, kernel_size=3, stride=1, padding=1)
        self.pr32 = Conv(ngf * 8, 1, kernel_size=3, stride=1, padding=1)
        self.pr16 = Conv(ngf * 4, 1, kernel_size=3, stride=1, padding=1)
        self.pr8 = Conv(ngf * 2, 1, kernel_size=3, stride=1, padding=1)
        self.pr4 = Conv(ngf * 1, 1, kernel_size=3, stride=1, padding=1)
        self.pr2 = Conv(ngf // 2, 1, kernel_size=4, stride=1, padding=2)
        self.pr1 = Conv(20, 1, kernel_size=5, stride=1, padding=2)

        ################ up
        self.upconv6 = TConv(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.upconv5 = TConv(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1)
        self.upconv4 = TConv(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1)
        self.upconv3 = TConv(ngf * 2, ngf * 1, kernel_size=4, stride=2, padding=1)
        self.upconv2 = TConv(ngf * 1, ngf // 2, kernel_size=4, stride=2, padding=1)
        self.upconv1 = TConv(ngf // 2, ngf // 4, kernel_size=4, stride=2, padding=1)

        ################ iconv?
        self.iconv6 = Conv(ngf * 16 - 1, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.iconv5 = Conv(769, ngf * 4, kernel_size=3, stride=1, padding=1)
        self.iconv4 = Conv(385, ngf * 2, kernel_size=3, stride=1, padding=1)
        self.iconv3 = Conv(193, ngf * 1, kernel_size=3, stride=1, padding=1)
        self.iconv2 = Conv(97, ngf // 2, kernel_size=3, stride=1, padding=1)

    def forward(self, input, hint):
        v = self.downH(hint)

        x1 = self.down1(input)
        x2 = self.down2(x1)
        x3 = self.down3(torch.cat([x2, v], 1))
        x4 = self.down4(x3)

        m = self.tunnel4(x4)

        x = self.up_to3(m)
        x = self.up_to2(torch.cat([x, x3], 1))
        x = self.up_to1(torch.cat([x, x2, v], 1))
        x = F.tanh(self.exit(torch.cat([x, x1], 1)))
        return x


class def_netD(nn.Module):
    def __init__(self, ndf=64):
        super(def_netD, self).__init__()

        sequence = [
            nn.Conv2d(4, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # 128
            nn.LeakyReLU(0.2, True),

            Tunnel(1, ndf, ndf),
            DResNeXtBottleneck(ndf, ndf * 2, 2),  # 64

            Tunnel(2, ndf * 2, ndf * 2),
            DResNeXtBottleneck(ndf * 2, ndf * 4, 2),  # 32

            Tunnel(3, ndf * 4, ndf * 4),
            DResNeXtBottleneck(ndf * 4, ndf * 8, 2),  # 16

            Tunnel(4, ndf * 8, ndf * 8),
            DResNeXtBottleneck(ndf * 8, ndf * 16, 2),  # 8

            Tunnel(2, ndf * 16, ndf * 16),
            DResNeXtBottleneck(ndf * 16, ndf * 32, 2),  # 4

            nn.Conv2d(ndf * 32, 1, kernel_size=4, stride=1, padding=0, bias=False)

        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

        # TODO: fix relu bug


def def_netF():
    vgg16 = M.vgg16()
    vgg16.load_state_dict(torch.load('vgg16-397923af.pth'))
    vgg16.features = nn.Sequential(
        *list(vgg16.features.children())[:9]
    )
    for param in vgg16.parameters():
        param.requires_grad = False
    return vgg16.features
