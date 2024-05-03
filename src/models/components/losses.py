import torch
import torch.nn.functional as F
from torch import nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
# https://discuss.pytorch.org/t/rmse-loss-function/16540/4
    
class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambd=5e-3, epsilon=1e-6): 
        super().__init__()
        self.lambd = lambd
        self.epsilon = epsilon
        
    def forward(self, z_a, z_b):
        N, D = z_a.size(0), z_a.size(1)
        z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0) + self.epsilon) 
        z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0) + self.epsilon)
        c = torch.matmul(z_a_norm.T, z_b_norm) / N
        c_diff = (c - torch.eye(D, device=z_a.device)).pow(2)
        # Only scale off-diagonal elements by lambda
        off_diagonal_indices = torch.ones_like(c) - torch.eye(D, device=z_a.device)
        loss = c_diff.sum() + self.lambd * (c_diff * off_diagonal_indices).sum()
        return loss
    
class NegativeCosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x1 = F.normalize(x1, p=2, dim=1)  # L2 normalization 
        x2 = F.normalize(x2, p=2, dim=1)
        cos_sim = -1 * F.cosine_similarity(x1, x2) 
        ##negative cosine similarity  equivalent normalized MSE loss (Grill et al., 2020)
        return cos_sim.mean()


