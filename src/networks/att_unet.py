#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
    Attention U-Net model  
    adapt from paper: https://arxiv.org/pdf/1804.03999.pdf  
    source: https://github.com/LeeJunHyun/Image_Segmentation
"""

__author__ = "Shiqi Hu"
__copyright__ = "Copyright 2022, Shiqi Hu"
__version__ = "1.0.0"
__email__ = "yj2682@columbia.edu"

import torch
import torch.nn as nn
from .blocks import DoubleConv, Attention_block, Up_Conv


class AttU_Net(nn.Module):
    def __init__(self, name, in_channels, out_channels):
        super(AttU_Net,self).__init__()
        self.in_channels = in_channels
        self.output_ch = out_channels
        self.name = name
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = DoubleConv(in_channels=self.in_channels, out_channels=32, dropout=0.1)
        self.Conv2 = DoubleConv(in_channels=32,out_channels=64, dropout=0.1)
        self.Conv3 = DoubleConv(in_channels=64,out_channels=128, dropout=0.2)
        self.Conv4 = DoubleConv(in_channels=128,out_channels=256, dropout=0.2)
        # self.Conv5 = DoubleConv(in_channels=256,out_channels=512, dropout=0.3)

        # self.Up5 = Up_Conv(in_channels=512,out_channels=256)
        # self.Att5 = Attention_block(F_g=256,F_l=256,F_int=128)
        # self.Up_Conv5 = DoubleConv(in_channels=512, out_channels=256, dropout=0.2)

        self.Up4 = Up_Conv(in_channels=256,out_channels=128)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_Conv4 = DoubleConv(in_channels=256, out_channels=128, dropout=0.2)
        
        self.Up3 = Up_Conv(in_channels=128,out_channels=64)
        self.Att3 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_Conv3 = DoubleConv(in_channels=128, out_channels=64, dropout=0.1)
        
        self.Up2 = Up_Conv(in_channels=64,out_channels=32, dropout=0.1)
        self.Att2 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.Up_Conv2 = DoubleConv(in_channels=64, out_channels=32, dropout=0.1)

        self.Conv_1x1 = nn.Conv2d(32,self.output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # #decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_Conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_Conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_Conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_Conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
