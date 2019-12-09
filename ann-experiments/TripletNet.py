import torch.nn as nn
import torch.nn.functional as F

from EmbeddingNet import EmbeddingNet

class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        self.embeddingnet = EmbeddingNet()

    def getEmbedding(self, x):
        return self.embeddingnet(x)
        
    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z