import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
        正卷积
    """
    def __init__(self, input_data, output_data):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data),
            nn.RReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(output_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data),
            nn.RReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DoubleConvBlock(nn.Module):
    """
    两层卷积
    """
    def __init__(self, in_data, out_data, kernel_size=3, stride=1, padding=1):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_data, out_data, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_data),
            nn.RReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_data, out_data, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_data),
            nn.RReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, input_data, output_data):
        super(ConvTransBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(input_data, output_data, kernel_size=3, stride=2, padding=1, output_padding=1,
                               dilation=1),
            nn.BatchNorm3d(output_data),
            nn.RReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, input_data, output_data, mode='V1'):
        super(UpBlock, self).__init__()
        self.up = ConvTransBlock(input_data, output_data)
        self.down = ConvBlock(2 * output_data, output_data)
        if mode == 'V1':
            self.down = nn.Sequential(
                ConvBlock(2 * output_data, 2 * output_data),
                ConvBlock(2 * output_data, output_data)
            )

    def forward(self, x, down_features):
        x = self.up(x)
        x = torch.cat([x, down_features], dim=1)  # 横向拼接
        x = self.down(x)
        return x


class ConvBlockWithKernel3(nn.Module):
    def __init__(self, input_data, output_data):
        super(ConvBlockWithKernel3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data),
            nn.RReLU(lower=0.125, upper=1.0 / 3, inplace=True)
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class NormalResBlock(nn.Module):
    def __init__(self, input_data, output_data, stride=2):
        super(NormalResBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(output_data),
            nn.RReLU(inplace=True),
            nn.Conv3d(output_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data)
        )
        self.conv = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(output_data)
        )
        self.relu = nn.RReLU(inplace=True)

    def forward(self, x):
        res = self.resblock(x)
        x = self.conv(x)
        x += res
        x = self.relu(x)
        return x


class NormalResUpBlock(nn.Module):
    def __init__(self, input_data, output_data):
        super(NormalResUpBlock, self).__init__()
        self.up = ConvTransBlock(input_data, output_data)
        self.down = nn.Sequential(
            NormalResBlock(2*output_data, output_data, stride=1),
            NormalResBlock(output_data, output_data, stride=1)
        )

    def forward(self, x, down_features):
        x = self.up(x)
        x = torch.cat([x, down_features], dim=1)
        x = self.down(x)
        return x


class ZeroResBlock(nn.Module):
    def __init__(self, in_data, out_data):
        super(ZeroResBlock, self).__init__()
        self.res1 = nn.Sequential(
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(in_data),
            nn.RReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(in_data)
        )
        self.act = nn.RReLU(inplace=False)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_data, out_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_data),
            nn.RReLU(inplace=True)
        )

    def forward(self, x):
        t = self.res1(x)
        x = self.act(x+t)
        x = self.conv1(x)
        return x


class RoConvBlock(nn.Module):
    """
        Head convolution.
    """
    def __init__(self, in_data, out_data):
        super(RoConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.Conv3d(in_data, out_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_data)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(out_data, out_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_data)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RoResBlock(nn.Module):
    """
    ReLU-only pre-activation
    """
    def __init__(self, in_data, out_data):
        super(RoResBlock, self).__init__()
        self.res = nn.Sequential(
            nn.RReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(in_data),
            nn.RReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(in_data)
        )
        self.conv = nn.Sequential(
            nn.RReLU(inplace=True),
            nn.Conv3d(in_data, out_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_data)
        )

    def forward(self, x):
        t = self.res(x)
        x = self.conv(t+x)
        return x


class FConvBlock(nn.Module):
    """
        Head convolution.
    """
    def __init__(self, in_data, out_data):
        super(FConvBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_data, out_data, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.BatchNorm3d(out_data),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_data, out_data, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FResBlock(nn.Module):
    """
    Full pre-activation
    """
    def __init__(self, in_data, out_data):
        super(FResBlock, self).__init__()
        self.res = nn.Sequential(
            nn.BatchNorm3d(in_data),
            nn.RReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(in_data),
            nn.RReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_data),
            nn.RReLU(inplace=True),
            nn.Conv3d(in_data, out_data, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        t = self.res(x)
        x = self.conv(t+x)
        return x

