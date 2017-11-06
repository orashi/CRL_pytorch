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


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


class FlowNetS(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(FlowNetS, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,
                                      0.02 / n)  # this modified initialization seems to work better, but it's very hacky
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2


def flownets(path=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)
    Args:
        path : where to load pretrained network. will create a new one if not set
    """
    model = FlowNetS(batchNorm=False)
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model