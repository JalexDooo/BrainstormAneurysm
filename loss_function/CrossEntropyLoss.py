import torch as t
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class CrossEntropyLoss(_Loss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, predict, target):
        return self.loss(predict, target.long())

