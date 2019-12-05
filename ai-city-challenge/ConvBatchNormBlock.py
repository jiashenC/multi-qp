import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBatchNormBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBatchNormBlock, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, 3)
        self.max_pooling = nn.MaxPool2d(2, 2)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.max_pooling(out)
        return out