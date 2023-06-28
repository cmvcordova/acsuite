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

class AutoEncoder(nn.Module):

    def __init__(
    self, 
    in_features: int = 1024, 
    code_features: int = 64, 
    hidden_layers: int = 2, 
    layer_activation: nn.Module = nn.ReLU(),
    output_activation: nn.Module = nn.Sigmoid(),
    dropout: float = 0.2,
    layer_features: np.ndarray = None):

        """
        Autoencoder for molecular data. Takes ECFP-like features as input and 
        outputs a compressed fingerprint as the autoencoder's latent code.

        Note: currently only works with canonical ECFP sizes e.g. 512, 1024, 2048, 4096
        and corresponding hidden layer combinations

        Args:
            in_features (int): Number of input features, Uses canonical ECFP Sizes e.g. 512, 1024, 2048, 4096
            code_features (int): Number of features in the code, i.e. size of the compressed fingerprint
            hidden_layers (int): Number of hidden layers
            layer_activation (nn.Module, optional): Activation function to use for the code. Defaults to nn.ReLU().
            output_activation (nn.Module, optional): Activation function to use for the output. Defaults to nn.Sigmoid().
            dropout (float, optional): Dropout probability. Defaults to 0.2.
            layer_features (np.ndarray, optional): Array specifying the number of features in each layer. Defaults to None, since it is calculated from the other parameters.
        """
        
        super().__init__()
        self.in_features = in_features
        self.code_features = code_features
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        ## array specifying the number of features in each layer if halving w.r.t former layer
        ## defaults to halving the input layer size for each hidden layer until the code layer
        ## e.g. 1024, 512, 256, 128, 64, can provide own as long as it meets assertions below
        self.layer_features = np.concatenate((
        self.in_features,
        [self.in_features//2**i for i in range(1, self.hidden_layers+2)],
        self.code_features),
        axis = None)

        assert self.in_features > self.code_features, f"Input features must be greater than code features: {in_features} !> {code_features}"
        assert self.layer_features[-2] > self.code_features, f"Final hidden layer output features must be greater than code features: {self.layer_features[-1]} !> {code_features}"

        ## build the encoder, specify the input layer separately to avoid activation, dropout in input layer
        self.encoder = nn.ModuleList([nn.Linear(self.layer_features[0], self.layer_features[1])])

        for i in range(1, len(self.layer_features)-1):
            self.encoder.append(nn.Linear(self.layer_features[i], self.layer_features[i+1]))
            if i == len(self.layer_features)-2: ## stop activation, dropout after inserting code layer
                break
            self.encoder.append(layer_activation)
            if dropout > 0.0:
                self.encoder.append(nn.Dropout(p=self.dropout))


        ## build the decoder
        self.decoder = nn.ModuleList()
        
        for i in range(len(self.layer_features)-1, 0, -1):
            self.decoder.append(nn.Linear(self.layer_features[i], self.layer_features[i-1]))

        ## initialize weights
        self.initialize_weights()

        ## unroll into sequential modules to avoid specifying ModuleList method in forward pass
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)

    def forward_encoder(self, x):
        z = self.encoder(x)
        return z
    
    def forward_decoder(self, z):
        recon_x = self.decoder(z)
        return recon_x
    
    def forward(self, x):
        z = self.forward_encoder(x)
        logits = self.forward_decoder(z)
        return logits
    
    def initialize_weights(self):
        ## using subordinate function in case we want to initialize weights differently
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        ## use kaiming for relu
        ## currently assumes all layers are activated with a ReLU
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if isinstance(m , nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

if __name__ == "__main__":
    _ = SiameseAutoEncoder()