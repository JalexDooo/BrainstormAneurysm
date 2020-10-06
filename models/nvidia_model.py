import torch as t
import torch.nn as nn


class BackboneEncoderBlock(nn.Module):
    def __init__(self, input_data, output_data, stride=1, padding=1, num_groups=8, activation='relu', normalization='group'):
        super(BackboneEncoderBlock, self).__init__()    
        if normalization == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=input_data)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=input_data)
        
        if activation == 'relu':
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == 'rrelu':
            self.actv1 = nn.RReLU(inplace=True)
            self.actv2 = nn.RReLU(inplace=True)

        self.conv1 = nn.Conv3d(in_channels=input_data, out_channels=output_data, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=input_data, out_channels=output_data, kernel_size=3, stride=stride, padding=padding)
    
    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.actv1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.actv2(x)
        x = self.conv2(x)

        x += res
        return x


class BackboneDecoderBlock(nn.Module):
    def __init__(self, input_data, output_data, stride=1, padding=1, num_groups=8, activation='relu', normalization='group'):
        super(BackboneDecoderBlock, self).__init__()    
        if normalization == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=output_data)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=output_data)
        
        if activation == 'relu':
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == 'rrelu':
            self.actv1 = nn.RReLU(inplace=True)
            self.actv2 = nn.RReLU(inplace=True)

        self.conv1 = nn.Conv3d(in_channels=input_data, out_channels=output_data, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=output_data, out_channels=output_data, kernel_size=3, stride=stride, padding=padding)
    
    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.actv1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.actv2(x)
        x = self.conv2(x)

        x += res
        return x


class BackboneDownSampling(nn.Module):
    def __init__(self, input_data, output_data, stride=2, kernel_size=3, padding=1, dropout_rate=None):
        super(BackboneDownSampling, self).__init__()
        self.dropout_flag = False
        self.conv1 = nn.Conv3d(in_channels=input_data, out_channels=output_data, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        if dropout_rate is not None:
            self.dropout_flag = True
            self.dropout = nn.Dropout3d(p=dropout_rate, inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        if self.dropout_flag:
            x = self.dropout(x)
        return x
