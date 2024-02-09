import torch
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
    def __init__(self, lambd=5e-3):
        super().__init__()
        self.lambd = lambd
        
    def forward(self, z_a, z_b):
        N, D = z_a.size(0), z_a.size(1)
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)
        c = torch.matmul(z_a_norm.T, z_b_norm) / N
        c_diff = (c - torch.eye(D, device=z_a.device)).pow(2)
        # Only scale off-diagonal elements by lambda
        off_diagonal_indices = torch.ones_like(c) - torch.eye(D, device=z_a.device)
        loss = c_diff.sum() + self.lambd * (c_diff * off_diagonal_indices).sum() - c_diff.trace()
        return loss
