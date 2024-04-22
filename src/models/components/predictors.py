import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from src.models.components.encoders import *
from src.models.encoding_module import ACAModule

class HotSwapEncoderMLP(nn.Module): 
    """
    MLP that optionally builds upon a pretrained encoder.
    Adjusts the provided encoder: freezes its weights and if the provided input features 
    are different from the encoder's, it adds a new input linear map layer that's the same 
    size as the input to the encoder. Else, returns an MLP. 
    """
    def __init__(self, 
            layer_activation: nn.Module = nn.ReLU(),
            in_features: int = 2048,
            hidden_features: int = 100, 
            hidden_layers: int = 2,
            output_features: int = 1, 
            dropout: float = 0.2,
            encoder: Optional[nn.Module] = None,
            pretrained_encoder_ckpt: Optional[str] = None 
            ):
   
            super().__init__()
            assert not (encoder is not None and pretrained_encoder_ckpt is not None), \
                "Either an encoder or a pretrained encoder checkpoint can be provided, not both."

            self.layer_activation = layer_activation
            self.in_features = in_features
            self.hidden_features = hidden_features
            self.hidden_layers = hidden_layers
            self.output_features = output_features
            self.dropout = dropout
            self.head = nn.Identity()
            mlp_input_features = in_features
            
            if pretrained_encoder_ckpt:
                encoder = self.load_pretrained_encoder(pretrained_encoder_ckpt)

            # Construct the head if an encoder is provided, adjusting input features if necessary
            if encoder:
                first_layer_in_features = self._get_first_layer_in_features(encoder)
                if in_features != first_layer_in_features:
                    encoder = nn.Sequential(nn.Linear(in_features, first_layer_in_features), encoder)
                mlp_input_features = self._get_last_layer_out_features(encoder)
                self.head = encoder

            # Construct the MLP
            mlp_layers = []
            for i in range(hidden_layers):
                mlp_layers.append(nn.Linear(mlp_input_features if i == 0 else hidden_features, hidden_features))
                mlp_layers.append(layer_activation)
                if dropout > 0:
                    mlp_layers.append(nn.Dropout(dropout))

            self.mlp = nn.Sequential(*mlp_layers)
            self.output_layer = nn.Linear(hidden_features, output_features)

    def forward(self, x):
        x = self.head(x)
        x = self.mlp(x)
        return self.output_layer(x)

    def _get_first_layer_in_features(self, module: nn.Module) -> int:
        for m in module.modules():
            if hasattr(m, 'in_features'):  
                return m.in_features
            elif isinstance(m, nn.Conv2d):  
                return m.in_channels
            # Extend with elif blocks for other types as necessary
        raise ValueError("Unsupported encoder structure: Cannot determine input features.")

    def _get_last_layer_out_features(self, module: nn.Module) -> int:
        for layer in reversed(list(module.children())):
            if isinstance(layer, nn.Linear):
                return layer.out_features
            else:
                out_features = self._get_last_layer_out_features(layer)
                if out_features is not None:
                    return out_features
        return None

    def load_pretrained_encoder(self, checkpoint_path: str) -> nn.Module:
        """
        Loads a pretrained encoder from the given checkpoint path into a Lightning ACAModule
        """
        if os.path.isfile(checkpoint_path):
            print(f"Loading pretrained encoder from {checkpoint_path}")
            pretrained_encoder = ACAModule.load_from_checkpoint(checkpoint_path=checkpoint_path)
            encoder = pretrained_encoder.net.encoder
            for param in encoder.parameters():
                param.requires_grad = False
            return encoder
        else:
            print(f"Pretrained encoder checkpoint not found at {checkpoint_path}.")
            return nn.Identity()



