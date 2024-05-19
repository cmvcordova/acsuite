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
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Compute the Euclidean distance between the two outputs
        distance = F.pairwise_distance(output1, output2)

        # Compute the contrastive loss
        loss = (1 - label) * 0.5 * torch.pow(distance, 2) + \
               label * 0.5 * torch.pow(F.relu(self.margin - distance), 2)

        return torch.mean(loss)

class SiamACLoss(nn.Module):
    def __init__(self, margin=1.0, difference_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.contrastive_loss = ContrastiveLoss(margin)
        self.difference_weight = difference_weight

    def forward(self, x1, recon_1, x2, recon_2, label):
        # Reconstruction Loss
        rec_loss1 = self.bce(recon_1, x1)
        rec_loss2 = self.bce(recon_2, x2)

        # Normalize the reconstructed vectors
        recon_1_norm = F.normalize(recon_1, p=2, dim=1)
        recon_2_norm = F.normalize(recon_2, p=2, dim=1)

        # Contrastive Loss
        contrastive_loss = self.contrastive_loss(recon_1_norm, recon_2_norm, label)

        return rec_loss1 + rec_loss2 + self.difference_weight * contrastive_loss
    
class NegativeCosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x1 = F.normalize(x1, p=2, dim=1)  # L2 normalization 
        x2 = F.normalize(x2, p=2, dim=1)
        cos_sim = -1 * F.cosine_similarity(x1, x2) 
        ##negative cosine similarity  equivalent normalized MSE loss (Grill et al., 2020)
        return cos_sim.mean()





