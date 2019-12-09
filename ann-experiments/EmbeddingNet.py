import torch.nn as nn
from ConvBatchNormBlock import ConvBatchNormBlock

class EmbeddingNet(nn.Module):

    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.conv_block1 = ConvBatchNormBlock(3, 16)
        self.conv_block2 = ConvBatchNormBlock(16, 32)
        self.conv_block3 = ConvBatchNormBlock(32, 64)
        self.conv_block4 = ConvBatchNormBlock(64, 128)
        self.conv_block5 = ConvBatchNormBlock(128, 256)
        self.flatten_adaptive_pooling = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.conv_block5(out)
        out = self.flatten_adaptive_pooling(out)
        out = out.view(batch_size, -1)
        return out