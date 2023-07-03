import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648

class AE_MLP(nn.Module):
    
        def __init__(self, 
            pretrained_autoencoder, out_features: int, hidden_features: int,
            activation: nn.Module = nn.ReLU(), dropout: float = 0.0):
    
            """
            MLP that builds upon a pretrained autoencoder. Removes the input layer and the decoder from the autoencoder,
            freezes its weights and adds a new input layer that's the same fingerprint size as the layers used to
            and an MLP that learns from the autoencoder's generated features.
    
            Args:
                in_features (int): Number of input features, i.e. size of the compressed fingerprint
                out_features (int): Number of output features, e.g. number of properties to predict
                hidden_features (int): Number of hidden features
                activation (nn.Module, optional): Activation function to use for the code. Defaults to nn.Sigmoid().
                dropout (float, optional): Dropout probability. Defaults to 0.0.
            """
            super().__init__()
    
            self.mlp = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_features, hidden_features),
                activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_features, out_features)
            )
    
        def forward(self, x):
            return self.mlp(x)

