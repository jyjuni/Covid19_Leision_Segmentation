#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
    Attention U-Net model    
"""

__author__ = "Yijia Jin"
__copyright__ = "Copyright 2022, Yijia Jin"
__version__ = "1.0.0"
__email__ = "yj2682@columbia.edu"

import torch
import torch.nn as nn
from .blocks import DoubleConv, Down, Up, OutConv, Attention_block


class AttU_Net(nn.Module):
    def __init__(self, name, n_channels, n_classes):
        super(AttU_Net,self).__init__()
        self.img_ch = n_channels
        self.output_ch = n_classes
        self.name = name
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = DoubleConv(in_channels=self.img_ch,out_channels=64)
        self.Conv2 = DoubleConv(in_channels=64,out_channels=128)
        self.Conv3 = DoubleConv(in_channels=128,out_channels=256)
        self.Conv4 = DoubleConv(in_channels=256,out_channels=512)
        self.Conv5 = DoubleConv(in_channels=512,out_channels=1024)

        self.Up5 = up_conv(in_channels=1024,out_channels=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = DoubleConv(in_channels=1024, out_channels=512)

        self.Up4 = up_conv(in_channels=512,out_channels=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = DoubleConv(in_channels=512, out_channels=256)
        
        self.Up3 = up_conv(in_channels=256,out_channels=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = DoubleConv(in_channels=256, out_channels=128)
        
        self.Up2 = up_conv(in_channels=128,out_channels=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = DoubleConv(in_channels=128, out_channels=64)

        self.Conv_1x1 = nn.Conv2d(64,self.output_ch,kernel_size=1,stride=1,padding=0)


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
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# class conv_block(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self,in_channels,out_channels):
#         super(conv_block,self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )


#     def forward(self,x):
#         x = self.conv(x)
#         return x

class up_conv(nn.Module):
  """Upscaling then double conv"""

  def __init__(self,in_channels, out_channels):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )

  def forward(self,x):
        x = self.up(x)
        return x

