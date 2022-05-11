#!/usr/bin/python
# -*- coding: utf-8 -*-

"""U-net architecture"""

__author__ = "Yijia Jin"
__copyright__ = "Copyright 2022, Yijia Jin"
__version__ = "1.0.0"
__email__ = "yj2682@columbia.edu"

import torch
import torch.nn as nn
from .blocks import DoubleConv, Down, Up, OutConv

class UNet_Reduced(nn.Module):
    def __init__(self, name, n_channels, n_classes):
        super(UNet_Reduced, self).__init__()
        self.name = name
        self.n_channels = n_channels
        self.n_classes = n_classes


        self.inputL = DoubleConv(n_channels, 32, 0.1)
        self.down1 = Down(32, 32, 0.1)
        self.down2 = Down(32, 64, 0.2)
        self.down3 = Down(64, 128, 0.2)
        self.down4 = Down(128, 256, 0.3)
        self.up1 = Up(256, 128, 0.2)
        self.up2 = Up(128, 64, 0.2)
        self.up3 = Up(64, 32, 0.1)
        self.up4 = Up(32, 32, 0.1)
        self.outputL = OutConv(32, n_classes)


    def forward(self, x):
        x1 = self.inputL(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b = self.down4(x4)
        
        x = self.up1(b, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.outputL(x)
        
        return x

class UNet(nn.Module):
    def __init__(self, name, n_channels, n_classes):
        super(UNet, self).__init__()
        self.name = name
        self.n_channels = n_channels
        self.n_classes = n_classes

        # self.inputL = DoubleConv(n_channels, 64, 0.1)
        # self.down1 = Down(64, 128, 0.1)
        # self.down2 = Down(128, 256, 0.2)
        # self.down3 = Down(256, 512, 0.2)
        # self.down4 = Down(512, 1024, 0.3)
        # self.up1 = Up(1024, 512, 0.2)
        # self.up2 = Up(512, 256, 0.2)
        # self.up3 = Up(256, 128, 0.1)
        # self.up4 = Up(128, 64, 0.1)
        # self.outputL = OutConv(64, n_classes)

        # baseline model with no dropouts
        self.inputL = DoubleConv(n_channels, 64, 0)
        self.down1 = Down(64, 128, 0)
        self.down2 = Down(128, 256, 0)
        self.down3 = Down(256, 512, 0)
        self.down4 = Down(512, 1024, 0)
        self.up1 = Up(1024, 512, 0)
        self.up2 = Up(512, 256, 0)
        self.up3 = Up(256, 128, 0)
        self.up4 = Up(128, 64, 0)
        self.outputL = OutConv(64, n_classes)


    def forward(self, x):
        x1 = self.inputL(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b = self.down4(x4)
        
        x = self.up1(b, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.outputL(x)
        
        return x
