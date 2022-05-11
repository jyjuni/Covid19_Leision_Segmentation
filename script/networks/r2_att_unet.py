
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv, Down, RRCNN_block, Up, OutConv, Attention_block, Up_Conv

"""
    R2Att-UNet model 
    integration of  R2U-Net and Attention U-Net
    adapt from: https://github.com/LeeJunHyun/Image_Segmentation
"""

class R2AttU_Net(nn.Module):
    def __init__(self,name, img_ch=3,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        self.name = name
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(in_channels=img_ch,out_channels=32,t=t)

        self.RRCNN2 = RRCNN_block(in_channels=32,out_channels=64,t=t)
        
        self.RRCNN3 = RRCNN_block(in_channels=64,out_channels=128,t=t)
        
        self.RRCNN4 = RRCNN_block(in_channels=128,out_channels=256,t=t)
        
        self.RRCNN5 = RRCNN_block(in_channels=256,out_channels=512,t=t)
        

        self.Up5 = Up_Conv(in_channels=512,out_channels=256)
        self.Att5 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN5 = RRCNN_block(in_channels=512, out_channels=256,t=t)
        
        self.Up4 = Up_Conv(in_channels=256,out_channels=128)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN4 = RRCNN_block(in_channels=256, out_channels=128,t=t)
        
        self.Up3 = Up_Conv(in_channels=128,out_channels=64)
        self.Att3 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN3 = RRCNN_block(in_channels=128, out_channels=64,t=t)
        
        self.Up2 = Up_Conv(in_channels=64,out_channels=32)
        self.Att2 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.Up_RRCNN2 = RRCNN_block(in_channels=64, out_channels=32,t=t)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1