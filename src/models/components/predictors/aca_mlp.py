import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.models.acamodule import ACAModule

#https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648

class ACA_MLP(nn.Module):
    
        def __init__(self, 
            pretrained_autoencoder_ckpt: str = None,
            activation: nn.Module = nn.ReLU(),
            in_features: int = 2048,
            out_features: int = 1, 
            hidden_features: int = 100, ## Ilnicka and Schneider 2023
            depth: int = 2,
            dropout: float = 0.0):
    
            """
            MLP that builds upon a pretrained autoencoder. Removes the input layer and the decoder from the autoencoder,
            freezes its weights and adds a new input layer that's the same fingerprint size as the layers used to
            and an MLP that learns from the autoencoder's generated features.

            Optionally just returns an MLP
     
            Args:
                pretrained_autoencoder_ckpt (str, optional): Path to the pretrained autoencoder checkpoint. Defaults to None.
                activation (nn.Module, optional): Activation function to use for the code. Defaults to nn.ReLU().
                in_features (int): Number of input features, i.e. size of the compressed fingerprint
                out_features (int): Number of output features, e.g. number of properties to predict
                hidden_features (int): Number of hidden features
                depth (int): Number of hidden layers                
                dropout (float, optional): Dropout probability. Defaults to 0.0.
            """
            super().__init__()
            pretrained_autoencoder_ckpt = 'data/models/autoencoders/ae_test.ckpt'
            if pretrained_autoencoder_ckpt is not None:
                encoder = ACAModule.load_from_checkpoint(checkpoint_path = pretrained_autoencoder_ckpt, map_location={'cuda:0':'cpu'}).net.encoder
                # remove the input layer and keep only the encoder
                self.encoder = nn.Sequential(*list(encoder.children())[1:])
            ## freeze the encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            ## add a new input layer that's the same fingerprint size as the layers used to
            self.encoder[0] = nn.Linear(in_features, self.encoder[0].out_features)
            ## add an MLP that learns from the autoencoder's generated features

            """
            self.mlp = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_features, hidden_features),
                activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_features, out_features)
            )
            """
    
        def forward(self, x):
            return self.mlp(x)


print(ACA_MLP())

if __name__ == "__main__":
    _ = ACA_MLP()