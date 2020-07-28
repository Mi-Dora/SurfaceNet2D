# -*- coding: utf-8 -*-
"""
    Created on Wednesday, Jul 15 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Wednesday, Jul 15 2020

Southeast University, College of Automation, 211189 Nanjing China
"""

import torch.nn as nn
import torch


class SurfaceNet2d(nn.Module):
    """
    SurfaceNet 2D accepts two CVCs as input
    Output the confidence of the voxel to be in the surface
    """
    def __init__(self):
        super(SurfaceNet2d, self).__init__()

        self.l1 = BnConvReLu2d(in_channels=6, out_channels=32, kernel_size=3, padding=1, max_pool=True)
        self.l2 = BnConvReLu2d(in_channels=32, out_channels=80, kernel_size=3, padding=1, max_pool=True)
        self.l3 = BnConvReLu2d(in_channels=80, out_channels=160, kernel_size=3, padding=1)
        self.l4 = BnConvReLu2d(in_channels=160, out_channels=300, kernel_size=3, padding=2, dilation=2)
        self.l5 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 100, kernel_size=3, padding=1),
            nn.BatchNorm2d(100),
            nn.Conv2d(100, 100, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(100)
        )
        self.s1 = BnUpConvSig2d(in_channels=32, out_channels=16, up_rate=2, kernel_size=1)
        self.s2 = BnUpConvSig2d(in_channels=80, out_channels=16, up_rate=4, kernel_size=1)
        self.s3 = BnUpConvSig2d(in_channels=160, out_channels=16, up_rate=4, kernel_size=1)
        self.s4 = BnUpConvSig2d(in_channels=300, out_channels=16, up_rate=4, kernel_size=1)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(100),
            nn.Conv2d(100, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, cvc1, cvc2):
        """
        :param cvc1: (tensor) size:(N, 3, s, s)
        :param cvc2: (tensor) size:(N, 3, s, s)
        :return: (tensor) size:(N, 1, s, s)
        """
        cvc = torch.cat((cvc1, cvc2), dim=1)  # (N, C, H, W)
        lo1 = self.l1(cvc)
        lo2 = self.l2(lo1)
        lo3 = self.l3(lo2)
        lo4 = self.l4(lo3)
        so1 = self.s1(lo1)
        so2 = self.s2(lo2)
        so3 = self.s3(lo3)
        so4 = self.s4(lo4)
        sum_so = so1 + so2 + so3 + so4
        sum_so = self.l5(sum_so)
        return self.output_layer(sum_so)


class BnConvReLu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, dilation=1, max_pool=False):
        super(BnConvReLu2d, self).__init__()
        self.max_pool = max_pool
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(out_channels)
        )
        if self.max_pool:
            self.MaxPooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        out = self.layer(x)
        if self.max_pool:
            out = self.MaxPooling(out)
        return out


class BnUpConvSig2d(nn.Module):
    def __init__(self, in_channels, out_channels, up_rate, kernel_size=1, padding=0):
        """
        :param in_channels: (int) number of input features
        :param out_channels: (int) number of output features
        :param up_rate: (int) output / input (side-length)
        :param kernel_size: (int or tuple)
        :param padding: (int)
        """
        super(BnUpConvSig2d, self).__init__()
        stride = up_rate
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=up_rate - 1),
            nn.Sigmoid()
        )
        # output_padding: padding for the output to make the dimension meet the requirement,
        # which could be computed through formula in
        # https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d

    def forward(self, x):
        # s_out = s_in * up_rate
        return self.layer(x)


if __name__ == '__main__':
    model = SurfaceNet2d()
    # random input data
    cvc1 = torch.rand(8, 3, 20, 20)
    cvc2 = torch.rand(8, 3, 20, 20)
    # CVC for 2D condition is not cube, but just colored image (plane)
    output = model(cvc1, cvc2)
    print(torch.cat((cvc1, cvc2), dim=1).shape)
    print(output.shape)


