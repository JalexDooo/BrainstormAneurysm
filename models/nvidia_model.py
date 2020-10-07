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


class VDResampling(nn.Module):
    def __init__(self, input_data=256, output_data=256, dense_features=(10, 12, 8), stride=2, kernel_size=3, padding=1, activation="relu", normalization="group"):
        super(VDResampling, self).__init__()        

        midput_data = input_data // 2
        self.dense_features = dense_features
        
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=input_data)
        if activation == 'relu':
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == 'rrelu':
            self.actv1 = nn.RReLU(inplace=True)
            self.actv2 = nn.RReLU(inplace=True)
        
        self.conv1 = nn.Conv3d(in_channels=input_data, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dense1 = nn.Linear(in_features=16*self.dense_features[0]*self.dense_features[1]*self.dense_features[2], out_features=input_data)
        self.dense2 = nn.Linear(in_features=midput_data, out_features=midput_data*self.dense_features[0]*self.dense_features[1]*self.dense_features[2])

        self.up0 = LinearUpSampling(midput_data, output_data)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features

    def forward(self, x):
        x = self.gn1(x)
        x = self.actv1(x)
        x = self.conv1(x)
        x = x.view(-1, self.num_flat_features(x))
        x_vd = self.dense1(x)
        distr = x_vd
        x = VDraw(x_vd)
        x = self.dense2(x)
        x = self.actv2(x)
        x = x.view((1, 128, self.dense_features[0], self.dense_features[1], self.dense_features[2]))
        x = self.up0(x)

        return x, distr


def VDraw(x):
    return t.distributions.Normal(x[:, :128], x[:, 128:]).sample()
