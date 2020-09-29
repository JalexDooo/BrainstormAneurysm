import torch as t
import torch.nn as nn


class DiscriminatorGAN(nn.Module):
    """MINIST Testing code.

    Criminator Network
    """
    def __init__(self):
        super(CriminatorGAN, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.dis(x)
        return x


class GenerativeGAN(nn.Module):
    def __init__(self, input_data):
        super(GenerativeGAN, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_data, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.gen(x)
        return x





