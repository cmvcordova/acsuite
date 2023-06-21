import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## CURRENTLY UNCHANGED W.R.T
### AUTOENCODER, MASKING TO COME SOON

class MaskedAutoEncoder(nn.Module):

    def __init__(self, in_features: int = 1024, code_features: int = 64,
    depth: int = 2, activation: nn.Module = nn.ReLU()):

    """
    Autoencoder for molecular data. Takes ECFP-like features as input and 
    outputs a compressed fingerprint as the autoencoder's code.

    Args:
        in_features (int): Number of input features, ECFP Size
        code_features (int): Number of features in the code, i.e. size of the compressed fingerprint
        activation (nn.Module, optional): Activation function to use for the code. Defaults to nn.Sigmoid().
    """
    super().__init__()

    assert in_features > code_features "Input features must be greater than code features"

    self.encoder = nn.Sequential(
        nn.Linear(in_features, in_features/2), # input layer
        activation,
        nn.Linear(in_features/2, in_features/4), ## hidden 1
        activation,
        nn.Linear(in_features/4, code_features), ## hidden 2, code output layer
    )
    
    self.decoder = nn.Sequential(
        nn.Linear(code_features, in_features/4), ## hidden 1,
        activation,
        nn.Linear(in_features/4, in_features/2), ## hidden 2
        activation,
        nn.Linear(in_features/2, in_features)
     ) ## output layer, restore original FP size

    def forward_encoder(self, x):
        z = self.encoder(x)
        return z
    
    def forward_decoder(self, z):
        x = self.decoder(z)
        return x
    
    def forward(self, x):
        z = self.forward_encoder(x)
        logits = self.forward_decoder(z)
        return logits
    
    #def initialize_weights(self):
        ## use kaiming for relu
