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


class LinearUpSampling(nn.Module):
    def __init__(self, input_data, output_data, scale_factor=2, mode='trilinear', align_corners=True):
        super(LinearUpSampling, self).__init__()        
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.conv1 = nn.Conv3d(in_channels=input_data, out_channels=output_data, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels=input_data, out_channels=output_data, kernel_size=1)
    
    def forward(self, x, skipx=None):
        x = self.conv1(x)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

        if skipx is not None:
            x = t.cat((x, skipx), 1)
            x = self.conv2(x)
        
        return x


class OutputTransition(nn.Module):
    def __init__(self, input_data, output_data):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=input_data, out_channels=output_data, kernel_size=1)
        self.actv1 = t.sigmoid
    
    def forward(self, x):
        return self.actv1(self.conv1(x))


class VDResampling(nn.Module):
    def __init__(self, input_data=256, output_data=256, dense_features=(10, 12, 8), stride=2, kernel_size=3, padding=1, activation="relu", normalization="group"):
        super(VDResampling, self).__init__()        

        midput_data = input_data // 2
        self.dense_features = dense_features
        # print('self.dense_features: {}'.format(self.dense_features))
        
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
        # print('VDResample: x: {}'.format(x.shape))
        x = x.view(-1, self.num_flat_features(x))
        # print('VDResample: view x: {}'.format(x.shape))
        # VDResample: x: torch.Size([1, 16, 9, 6, 9])
        # VDResample: view x: torch.Size([1, 7776])
        x_vd = self.dense1(x)
        distr = x_vd
        x = VDraw(x_vd)
        x = self.dense2(x)
        x = self.actv2(x)
        x = x.view((1, 128, self.dense_features[0], self.dense_features[1], self.dense_features[2]))
        x = self.up0(x)

        return x, distr


def VDraw(x):
    """Generate a Gaussian distribution with the given mean(128-d) and std(128-d)"""
    return t.distributions.Normal(x[:, :128], x[:, 128:]).sample()


class VAEDecoderBlock(nn.Module):
    def __init__(self, input_data, output_data, activation='relu', normalization='group', mode='trilinear'):
        super(VAEDecoderBlock, self).__init__()
        self.up0 = LinearUpSampling(input_data, output_data, mode=mode)
        self.block = BackboneDecoderBlock(output_data, output_data, activation=activation, normalization=normalization)
    
    def forward(self, x):
        x = self.up0(x)
        x = self.block(x)

        return x


class VAE(nn.Module):
    def __init__(self, input_data=256, output_data=4, dense_features=(10, 12, 8), activation='relu', normalization='group', mode='trilinear'):
        super(VAE, self).__init__()

        self.vd_resample = VDResampling(input_data, input_data, dense_features)
        self.vd_block2 = VAEDecoderBlock(input_data, input_data//2)
        self.vd_block1 = VAEDecoderBlock(input_data//2, input_data//4)
        self.vd_block0 = VAEDecoderBlock(input_data//4, input_data//8)
        self.vd_end = nn.Conv3d(input_data//8, output_data, kernel_size=1)
    
    def forward(self, x):
        # print('vd_resample before -> x: {}'.format(x.shape))
        # vd_resample before -> x: torch.Size([1, 256, 18, 12, 18])
        x, distr = self.vd_resample(x)
        # print('vd_resample after -> x: {}, distr: {}'.format(x.shape, distr.shape))
        x = self.vd_block2(x)
        x = self.vd_block1(x)
        x = self.vd_block0(x)
        x = self.vd_end(x)

        return x, distr


class NvNet(nn.Module):
    def __init__(self, input_data=4, output_data=4, activation='rrelu', normalization='group', mode='trilinear', vae_flag=True, shape=[144, 96, 144]):
        super(NvNet, self).__init__()
        self.shape = shape
        self.vae_flag = vae_flag

        # encoder blocks
        self.encode1 = nn.Sequential(
            BackboneDownSampling(input_data, 32, 1, dropout_rate=0.2),
            BackboneEncoderBlock(32, 32, activation=activation, normalization=normalization)
        )

        self.encode2 = nn.Sequential(
            BackboneDownSampling(32, 64),
            BackboneEncoderBlock(64, 64, activation=activation, normalization=normalization),
            BackboneEncoderBlock(64, 64, activation=activation, normalization=normalization)
        )

        self.encode3 = nn.Sequential(
            BackboneDownSampling(64, 128),
            BackboneEncoderBlock(128, 128, activation=activation, normalization=normalization),
            BackboneEncoderBlock(128, 128, activation=activation, normalization=normalization)
        )
        
        self.encode4 = nn.Sequential(
            BackboneDownSampling(128, 256),
            BackboneEncoderBlock(256, 256, activation=activation, normalization=normalization),
            BackboneEncoderBlock(256, 256, activation=activation, normalization=normalization),
            BackboneEncoderBlock(256, 256, activation=activation, normalization=normalization),
            BackboneEncoderBlock(256, 256, activation=activation, normalization=normalization)
        )

        # decoder blocks
        self.de_up2 = LinearUpSampling(256, 128, mode=mode)
        self.de_block2 = BackboneDecoderBlock(128, 128, activation=activation, normalization=normalization)
        self.de_up1 = LinearUpSampling(128, 64, mode=mode)
        self.de_block1 = BackboneDecoderBlock(64, 64, activation=activation, normalization=normalization)
        self.de_up0 = LinearUpSampling(64, 32, mode=mode)
        self.de_block0 = BackboneDecoderBlock(32, 32, activation=activation, normalization=normalization)


        self.decode_out = OutputTransition(32, output_data)

        # VAE encoder
        if self.vae_flag:
            self.dense_features = (shape[0]//16, shape[1]//16, shape[2]//16)
            self.vae = VAE(256, input_data, dense_features=self.dense_features)

    def forward(self, x):
        x1 = self.encode1(x)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)

        # print('x3.shape: {}, x4.shape: {}'.format(x3.shape, x4.shape))
        # x3.shape: torch.Size([1, 256, 18, 24, 18]), x4.shape: torch.Size([1, 256, 18, 24, 18])

        x3 = self.de_up2(x4, x3)
        x3 = self.de_block2(x3)

        x2 = self.de_up1(x3, x2)
        x2 = self.de_block1(x2)

        x1 = self.de_up0(x2, x1)
        x1 = self.de_block0(x1)

        x = self.decode_out(x1)
        # print('x.shape: {}'.format(x.shape))

        if self.vae_flag:
            vae, distr = self.vae(x4)
            print('vae.shape: {}, distr.shape: {}'.format(vae.shape, distr.shape))
            x = t.cat((x, vae), 1)
            return x, distr

        return x
