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

class AttentionUNet(torch.nn.Module):

    def __init__(self):
        super(AttentionUNet, self).__init__()

        # self.linear1 = torch.nn.Linear(100, 200)
        # self.activation = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(200, 10)
        # self.softmax = torch.nn.Softmax()

    def forward(self, x):
        # x = self.linear1(x)
        # x = self.activation(x)
        # x = self.linear2(x)
        # x = self.softmax(x)
        return x