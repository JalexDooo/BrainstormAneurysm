import torch as t
import torch.nn as nn
import math


class ConvBlock(nn.Module):
    def __init__(self, in_data, out_data, kernel=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_data, out_data, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm3d(out_data),
            nn.RReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_data, out_data, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm3d(out_data),
            nn.RReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvUpBlock(nn.Module):
    def __init__(self, in_data, out_data, kernel=3, stride=1, padding=1):
        super(ConvUpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_data, out_data, kernel_size=3, stride=2, padding=1, output_padding=1,
                               dilation=1),
            nn.BatchNorm3d(out_data),
            nn.RReLU(inplace=True),
        )
        self.down = ConvBlock(2*out_data, out_data)
    
    def forward(self, x, down_features):
        x = self.up(x)
        x = t.cat([x, down_features], dim=1)
        x = self.down(x)
        return x

class ResConvBlock_R0(nn.Module):
    def __init__(self, in_data, out_data, kernel=3, stride=1, padding=1):
        super(ResConvBlock_R0, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_data, out_data, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm3d(out_data),
            nn.RReLU(inplace=True),
            nn.Conv3d(out_data, out_data, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm3d(out_data)
        )
        self.res = nn.Sequential(
            nn.Conv3d(in_data, out_data, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm3d(out_data)
        )
        self.relu = nn.RReLU(inplace=True)

    def forward(self, x):
        res = self.res(x)
        x = self.conv(x)
        x += res
        x = self.relu(x)
        return x


