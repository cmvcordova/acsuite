import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from src.models.acamodule import ACAModule
from src.models.components.encoders import *

class HotSwapEncoderMLP(nn.Module): 
        def __init__(self, 
            layer_activation: nn.Module = nn.ReLU(),
            in_features: int = 2048,
            hidden_features: int = 100, 
            hidden_layers: int = 2,
            output_features: int = 1, 
            dropout: float = 0.2,
            pretrained_encoder_ckpt: Optional[str] = 'src/models/pretrained/encoders/yourencoder.ckpt'
            ):
            """
            MLP that optionally builds upon a pretrained encoder.
            Adjusts the provided encoder: freezes its weights and if the provided input features 
            are different from the encoder's, it adds a new input linear map layer that's the same 
            size as the input to the encoder. Else, returns an MLP. 
            """
            super().__init__()
            self.in_features = in_features
            self.hidden_features = hidden_features
            self.hidden_layers = hidden_layers
            self.output_features = output_features
            self.dropout = dropout
            self.pretrained_encoder_ckpt = pretrained_encoder_ckpt
            ## Instantiate the input layer:
            ## Map the input to the encoder's size with a linear layer if there's a mismatch
            ## else, use an identity layer.
            self.input_layer = nn.Identity()
            self.encoder = nn.Identity()

            mlp_input = self.in_features
            if self.pretrained_encoder_ckpt is not None:
                # Load the pretrained encoder
                pretrained_encoder = ACAModule.load_from_checkpoint(checkpoint_path = pretrained_encoder_ckpt, map_location = 'cpu')                
                self.encoder = pretrained_encoder.net.encoder
                for param in self.encoder.parameters():
                    param.requires_grad = False
                ## create input layer with out features equal to the in features of the 
                ## first layer of the pretrained encoder, to ensure compatibility
                if in_features != self.encoder[0].in_features:
                    self.input_layer = nn.Linear(in_features, pretrained_encoder.net.encoder[0].in_features)
                    mlp_input = self.encoder[-1].out_features ## set the input size of the MLP to the output size of the encoder
            ## Instantiate the predictor's (MLP) hidden layers
            mlp_layers = []
            for i in range(hidden_layers):
                mlp_layers.append(nn.Linear(mlp_input if i == 0 else hidden_features, hidden_features))
                mlp_layers.append(layer_activation)
                if dropout > 0:
                    mlp_layers.append(nn.Dropout(dropout))

            self.mlp = nn.Sequential(*mlp_layers)
            self.output_layer = nn.Linear(self.hidden_features, self.output_features)

        def forward(self, x):
            x = self.input_layer(x)
            x = self.encoder(x)
            x = self.mlp(x)
            x = self.output_layer(x)
            return x