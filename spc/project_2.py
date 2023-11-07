import numpy as np
import torch
from torch import nn
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # self.mlp_1=nn.Sequential(nn.Linear(128,256))
        # self.mlp_2 = nn.Sequential(nn.Linear(256, 128))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        avg_out = self.se(avg_result)
        # max_out = self.se(max_result)
        # avg_result=torch.squeeze(avg_result,3)
        # avg_result = torch.squeeze(avg_result, 2)
        # avg_out = self.mlp_1(avg_result)
        # avg_out = self.mlp_2(avg_out)
        # avg_out=torch.unsqueeze(avg_out,2)
        # avg_out=torch.unsqueeze(avg_out,3)
        output = self.sigmoid( avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)

        output = self.conv(avg_result )
        output = self.sigmoid(output)
        return output


class SPattenion_block(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.pw=nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=1,stride=1)
        self.dw=nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=7,padding=3,stride=1)
        self.act=nn.ReLU()



    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        # out=x
        out =x * self.ca(x)
        out1=out+residual
        out=self.pw(out1)
        # x=self.sa(out)
        out = out * self.sa(out)+out1
        out=self.dw(out)
        out=self.act(out)
        out=out

        return out



if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    kernel_size = input.shape[2]
    cbam =  SPattenion_block(channel=512, reduction=16, kernel_size=kernel_size)
    output = cbam(input)
    print(output.shape)
