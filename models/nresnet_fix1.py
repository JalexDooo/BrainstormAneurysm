import torch as t
import torch.nn as nn
import math

from .basicmodule import *


class NormalResNetFix1(nn.Module):
    def __init__(self, in_data=4, out_data=5):
        super(NormalResNetFix1, self).__init__()
        kn = [16, 32, 64, 128, 256]

        self.in_model = nn.Sequential(
            NormalResBlock(in_data, kn[0], stride=1),
            NormalResBlock(kn[0], kn[0], stride=1)
        )

        self.layerin1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            NormalResBlock(kn[0], kn[1], stride=1),
            NormalResBlock(kn[1], kn[1], stride=1)
        )
        self.layer1_1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            NormalResBlock(kn[1], kn[2], stride=1),
            NormalResBlock(kn[2], kn[2], stride=1)
        )
        self.layer1_2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            NormalResBlock(kn[2], kn[3], stride=1),
            NormalResBlock(kn[3], kn[3], stride=1)
        )
        self.layer1_3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            NormalResBlock(kn[3], kn[4], stride=1),
            NormalResBlock(kn[4], kn[4], stride=1)
        )
        self.up1 = NormalResUpBlock(kn[4], kn[3])
        self.up2 = NormalResUpBlock(kn[3], kn[2])
        self.up3 = NormalResUpBlock(kn[2], kn[1])
        self.up4 = NormalResUpBlock(kn[1], kn[0])

        self.out = NormalResBlock(kn[0], out_data, stride=1)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, ind):
        x1 = self.in_model(ind)

        x2 = self.layerin1(x1)
        x3 = self.layer1_1(x2)
        x4 = self.layer1_2(x3)
        x5 = self.layer1_3(x4)


        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.out(x)

        return x

