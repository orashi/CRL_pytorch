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
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, bias=False)

    def forward(self, x):
        return F.leaky_relu(self.conv.forward(x), negative_slope=0.1, inplace=True)


class CorrelationLayer(nn.Module):
    def __init__(self, args=None, padding=20, kernel_size=1, max_displacement=20, stride_1=1, stride_2=2):
        super(CorrelationLayer, self).__init__(args)
        self.pad = padding
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride_1 = stride_1
        self.stride_2 = stride_2

    def forward(self, x_1, x_2):
        """
        Arguments
        ---------
        x_1 : 4D torch.Tensor (bathch channel height width)
        x_2 : 4D torch.Tensor (bathch channel height width)
        """
        x_1 = x_1.transpose(1, 2).transpose(2, 3)
        x_2 = F.pad(x_2, tuple([self.pad for _ in range(4)])).transpose(1, 2).transpose(2, 3)
        mean_x_1 = torch.mean(x_1, 3)
        mean_x_2 = torch.mean(x_2, 3)
        sub_x_1 = x_1.sub(mean_x_1.expand_as(x_1))
        sub_x_2 = x_2.sub(mean_x_2.expand_as(x_2))
        st_dev_x_1 = torch.std(x_1, 3)
        st_dev_x_2 = torch.std(x_2, 3)

        out_vb = torch.zeros(1)
        _y = 0
        _x = 0
        while _y < self.max_displacement * 2 + 1:
            while _x < self.max_displacement * 2 + 1:
                c_out = (torch.sum(sub_x_1 * sub_x_2[:, _x:_x + x_1.size(1),
                                             _y:_y + x_1.size(2), :], 3) /
                         (st_dev_x_1 * st_dev_x_2[:, _x:_x + x_1.size(1),
                                       _y:_y + x_1.size(2), :])).transpose(2, 3).transpose(1, 2)
                out_vb = torch.cat((out_vb, c_out), 1) if len(out_vb.size()) != 1 else c_out
                _x += self.stride_2
            _y += self.stride_2
        return out_vb


class DispFulNet(nn.Module):
    def __init__(self, ngf=64):
        super(DispFulNet, self).__init__()

        ################ down
        self.conv1 = Conv(3, ngf, kernel_size=7, stride=2, padding=3)
        self.conv2 = Conv(ngf, ngf * 2, kernel_size=5, stride=2, padding=2)

        self.corr = nn.Sequential(nn.Conv2d(ngf * 4, 81, kernel_size=1, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv_rdi = nn.Sequential(nn.Conv2d(ngf * 2, ngf, kernel_size=1, stride=1, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv3 = Conv(145, ngf * 4, kernel_size=5, stride=2, padding=2)
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
        nn.Linear(7 * 7 * 64, 1024, bias=True)
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

    def forward(self, left, right):
        conv1a = self.conv1(left)
        conv1b = self.conv1(right)
        conv2a = self.conv2(left)
        conv2b = self.conv2(right)
        corr = self.corr(torch.cat([conv2a, conv2b], 1))
        conv_rdi = self.conv_rdi(conv2a)
        conv3 = self.conv3(torch.cat([corr, conv_rdi], 1))
        conv3_1 = self.conv3_1(conv3)
        conv4 = self.conv4(conv3_1)
        conv4_1 = self.conv4_1(conv4)
        conv5 = self.conv5(conv4_1)
        conv5_1 = self.conv5_1(conv5)
        conv6 = self.conv6(conv5_1)
        conv6_1 = self.conv6_1(conv6)

        pr_64 = self.pr64(conv6_1)
        upconv6 = self.upconv6(conv6_1)
        iconv6 = self.iconv6(torch.cat([upconv6, conv5_1, pr_64], 1))

        pr_32 = self.pr32(iconv6)
        upconv5 = self.upconv5(iconv6)
        iconv5 = self.iconv5(torch.cat([upconv5, conv4_1, pr_32], 1))

        pr_16 = self.pr16(iconv5)
        upconv4 = self.upconv4(iconv5)
        iconv4 = self.iconv4(torch.cat([upconv4, conv3_1, pr_16], 1))

        pr_8 = self.pr8(iconv4)
        upconv3 = self.upconv3(iconv4)
        iconv3 = self.iconv3(torch.cat([upconv3, conv2a, pr_8], 1))

        pr_4 = self.pr4(iconv3)
        upconv2 = self.upconv2(iconv3)
        iconv2 = self.iconv2(torch.cat([upconv2, conv1a, pr_4], 1))

        pr_2 = self.pr2(iconv2)
        upconv1 = self.upconv1(iconv2)

        pr_1 = self.pr1(torch.cat([upconv1, left, pr_2]))

        return pr_1
