import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, embeddingnet):
        super(SiameseNetwork, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x1, x2):
        output1 = self.embeddingnet(x1)
        output2 = self.embeddingnet(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embeddingnet(x)