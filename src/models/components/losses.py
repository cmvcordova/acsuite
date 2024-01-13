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
    ## check for correctness
    def __init__(self, lambd=5e-3, scale_loss=True):
        super().__init__()
        self.lambd = lambd
        self.scale_loss = scale_loss
        
    def forward(self, z_a, z_b):
        N = z_a.size(0)
        z_a = z_a - z_a.mean(0)
        z_b = z_b - z_b.mean(0)
        c = (1/N) * torch.matmul(z_a.T, z_b)
        c = c**2
        c = c * (1-torch.eye(N, device=z_a.device))
        if self.scale_loss:
            c = c / c.sum()
        loss = ((c - self.lambd)**2).sum()
        return loss