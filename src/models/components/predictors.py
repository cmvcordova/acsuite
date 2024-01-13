import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from src.models.acamodule import ACAModule
from src.models.components.encoders import JointAutoEncoder

#https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648

class ACA_MLP(nn.Module):
    
        def __init__(self, 
            layer_activation: nn.Module = nn.ReLU(),
            in_features: int = 2048,
            hidden_features: int = 100, 
            hidden_layers: int = 2, ## Ilnicka and Schneider 2023
            output_features: int = 1, 
            dropout: float = 0.0,
            pretrained_encoder_ckpt: Optional[str] = 'src/models/pretrained/encoders/last.ckpt'
            ):
    
            """
            MLP that builds upon a pretrained encoder. Removes the input layer and the decoder from the autoencoder,
            freezes its weights and adds a new input layer that's the same fingerprint size as the layers used to
            and an MLP that learns from the encoder's generated features.

            Optionally just returns an MLP
     
            Args:
                pretrained_autoencoder_ckpt: Path to the pretrained autoencoder checkpoint.
                layer_activation: Activation function to use for the code.
                in_features: Number of input features, i.e. size of the compressed fingerprint
                out_features: Number of output features, e.g. number of properties to predict
                hidden_features: Number of hidden features
                depth: Number of hidden layers                
                dropout: Dropout probability.
            """
            super().__init__()
            self.pretrained_encoder_ckpt = pretrained_encoder_ckpt
            self.in_features = in_features
            self.hidden_features = hidden_features
            self.hidden_layers = hidden_layers
            self.output_features = output_features
            self.dropout = dropout
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ## Instantiate the input block:
            ## Linear layer mapping the input to the hidden layer size if no
            ## pretrained autoencoder is supplied, otherwise the input block
            ## integrates the pretrained autoencoder's encoder
            if self.pretrained_encoder_ckpt is not None:
            ## todo: add full flag support consistent with trainer accelerator flag from config
                pretrained_encoder = ACAModule.load_from_checkpoint(checkpoint_path = pretrained_encoder_ckpt,
                                                                    map_location = self.device)
                #pretrained_encoder = ACAModule.load_from_checkpoint(checkpoint_path = pretrained_encoder_ckpt, 
                #                                                   map_location={'cuda:0':'cpu'})
                
            ## create input layer based on out features of first layer, to ensure
                ## compatibility with first hidden layer of the pretrained encoder
                self.input_block = nn.ModuleList(pretrained_encoder.net.encoder)
                # freeze the previous encoder's layers
                for param in self.input_block.parameters():
                    param.requires_grad = False
                ## add new input layer with consistent in_feature dims
                self.input_block[0] = nn.Linear(self.in_features, 
                                                self.input_block[0].out_features)
            else:
                self.input_block = nn.ModuleList([nn.Linear(self.in_features, self.hidden_features)])

            self.input_block = nn.Sequential(*self.input_block)

            ## Instantiate the predictor's (MLP) hidden layers for learning on the input block's output
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
            self.mlp = nn.Sequential(*self.mlp)

            ## add the output layer, regression or classification
            ##todo: add support for multiple classification classes/regression targets?
            ## is this not already supported by the output_features parameter?

            self.output_layer=(nn.Linear(self.hidden_features, self.output_features))

    
        def forward(self, x):
            x = self.input_block(x)
            x = self.mlp(x)
            x = torch.flatten(x, start_dim=1)
            x = self.output_layer(x)
            return x