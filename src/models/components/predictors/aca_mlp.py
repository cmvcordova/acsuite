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
            layer_activation: nn.Module = nn.ReLU(),
            in_features: int = 2048,
            hidden_features: int = 100, 
            hidden_layers: int = 2, ## Ilnicka and Schneider 2023
            output_features: int = 1, 
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
            self.pretrained_autoencoder_ckpt = pretrained_autoencoder_ckpt
            self.in_features = in_features
            self.hidden_features = hidden_features
            self.hidden_layers = hidden_layers
            self.output_features = output_features
            self.dropout = dropout

            ## Instantiate the input block:
            ## Linear layer mapping the input to the hidden layer size if no
            ## pretrained autoencoder is supplied, otherwise the input block
            ## integrates the pretrained autoencoder's encoder
            
            if self.pretrained_autoencoder_ckpt is not None:
                frozen_encoder = ACAModule.load_from_checkpoint(checkpoint_path = pretrained_autoencoder_ckpt, map_location={'cuda:0':'cpu'})
                frozen_encoder = nn.ModuleList(frozen_encoder.net.encoder[1:])
                # freeze the encoder's layers
                for param in frozen_encoder.parameters():
                    param.requires_grad = False
                self.input_block = nn.ModuleList([nn.Linear(self.in_features, frozen_encoder[0].in_features)])
                self.input_block.extend(frozen_encoder)
            else:
                self.input_block = nn.ModuleList([nn.Linear(self.in_features, self.hidden_features)])

            self.input_block = nn.Sequential(*self.input_block)

            ## Instantiate the MLP'S hidden layers for learning on the input block's output
            ## if no pretrained autoencoder is supplied, the input block
            ## is just a linear mapping input layer to the MLP

            self.mlp = nn.ModuleList([nn.Linear(self.input_block[-1].out_features, self.hidden_features)])
            self.mlp.append(layer_activation)
            
            ## variably instantiate the hidden layers
            for i in range(self.hidden_layers):
                self.mlp.append(nn.Linear(self.hidden_features, self.hidden_features))
                self.mlp.append(layer_activation)
                if self.dropout > 0:
                    self.mlp.append(nn.Dropout(self.dropout))
            ## add the output layer
            self.mlp.append(nn.Linear(self.hidden_features, self.output_features))

            self.mlp = nn.Sequential(*self.mlp)

    
        def forward(self, x):
            x = self.input_block(x)
            x = self.mlp(x)
            return x


print(ACA_MLP())

if __name__ == "__main__":
    _ = ACA_MLP()