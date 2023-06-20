import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseAutoEncoder(nn.Module):

    def __init__(self, in_features: int, code_features:int,
        activation: nn.Module = nn.Sigmoid()):

    """
    Autoencoder for molecular data. Takes ECFP-like features as input and 
    outputs a compressed fingerprint as the autoencoder's code.

    Args:
        in_features (int): Number of input features, ECFP Size
        code_features (int): Number of features in the code, i.e. size of the compressed fingerprint
        activation (nn.Module, optional): Activation function to use for the code. Defaults to nn.Sigmoid().
    """
    super().__init__()

    self.encoder = nn.Sequential(
        nn.Linear(in_features, code_features),
        activation,
        nn.Linear(code_features, code_features)
    )
    self.decoder = nn.Linear(code_features, in_features) ## restore original FP size

    def forward(self, x):
        _x = self.encoder(x)
        logits = self.decoder(_x)
        return logits