import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class JointAutoEncoder(nn.Module):

    def __init__(
    self, 
    in_features: int = 4096, 
    code_features: int = 256,  
    layer_activation: nn.Module = nn.ReLU(),
    output_activation: nn.Module = nn.Sigmoid(),
    dropout: float = 0.2):
        
        """
        Autoencoder for molecular data. Takes ECFP-like features as input and 
        outputs a compressed fingerprint as the autoencoder's latent code.

        Note: currently only works with canonical ECFP sizes e.g. 512, 1024, 2048, 4096
        and corresponding hidden layer combinations

        Args:
            in_features: Number of input features, Uses canonical ECFP Sizes e.g. 512, 1024, 2048, 4096
            code_features: Number of features in the code, i.e. size of the compressed fingerprint
            layer_activation: Activation function to use for the code. Defaults to nn.ReLU().
            output_activation: Activation function to use for the output. Defaults to nn.Sigmoid().
            dropout: Dropout probability. Defaults to 0.2.
        """
        
        super().__init__()
        
        self.in_features = in_features
        self.code_features = code_features
        assert self.in_features % self.code_features == 0, f"Input features must be evenly divisible by code features: {in_features} % {code_features} != 0"
        assert self.in_features > self.code_features, f"Input features must be greater than code features: {self.in_features} !> {self.code_features}"

        ## calculate layer features from in and code features, halving w.r.t former layer until code layer dimensions are obtained
        ## e.g. 1024, 512, 256, 128, 64, can provide own dimensions as long as it meets assertions below

        self.layer_features = []
        while in_features >= code_features: self.layer_features.append(in_features) ; in_features//=2  

        assert self.layer_features[-2] > self.code_features, f"Final hidden layer output features must be greater than code features: {self.layer_features[-1]} !> {code_features}"

        self.dropout = dropout


        ## build the encoder, specify the input layer separately to avoid activation, dropout in input layer
        self.encoder = nn.ModuleList()

        for i in range(0, len(self.layer_features)-1):
            self.encoder.append(nn.Linear(self.layer_features[i], self.layer_features[i+1]))
            if i == len(self.layer_features)-2: ## stop activation, dropout after inserting code layer
                break
            self.encoder.append(layer_activation)
            if dropout > 0.0:
                self.encoder.append(nn.Dropout(p=self.dropout))

        self.decoder = nn.ModuleList()

        ## build the decoder
        for i in range(len(self.layer_features)-1, 0, -1):
            self.decoder.append(nn.Linear(self.layer_features[i], self.layer_features[i-1]))
            if i == 1: ## stop activation, dropout before final layer
                break
            self.decoder.append(layer_activation)
            if dropout > 0.0:
                self.decoder.append(nn.Dropout(p=self.dropout))

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

class JointEncoder(nn.Module):

    def __init__(
    self, 
    in_features: int = 4096, 
    code_features: int = 256, 
    layer_activation: nn.Module = nn.ReLU(),
    output_activation: nn.Module = nn.Sigmoid(),
    training_type: Literal['binary_classification'] = 'binary_classification',
    dropout: float = 0.2):

        """
        Joint encoder for molecular data pre-trained on a binary classification objective (AC vs non AC). 
        Takes ECFP-like features as input. Final prediction layer is included only for training purposes.

        Note: currently only works with canonical ECFP sizes e.g. 512, 1024, 2048, 4096
        and corresponding hidden layer combinations

        Args:
            in_features: Number of input features, Uses canonical ECFP Sizes e.g. 512, 1024, 2048, 4096
            code_features: Number of features in the code, i.e. size of the compressed fingerprint
            layer_activation: Activation function to use for the code.
            output_activation: Activation function to use for the output. Currently unused.
            training_type: Training objective for the encoder. Currently supports binary classification.
            dropout: Dropout probability.
        """
        
        super().__init__()

        self.in_features = in_features
        self.code_features = code_features

        assert self.in_features % self.code_features == 0, f"Input features must be evenly divisible by code features: {in_features} % {code_features} != 0"
        assert self.in_features > self.code_features, f"Input features must be greater than code features: {in_features} !> {code_features}"
    
        ## calculate layer features from in and code features, halving w.r.t former layer until code layer dimensions are obtained
        ## e.g. 1024, 512, 256, 128, 64, can provide own dimensions as long as it meets assertions below
        self.layer_features = []
        while in_features >= code_features: self.layer_features.append(in_features) ; in_features//=2  

        assert self.layer_features[-2] > self.code_features, f"Final hidden layer output features must be greater than code features: {self.layer_features[-1]} !> {code_features}"
        
        self.dropout = dropout

        ## build the encoder, specify the input layer separately to avoid activation, dropout in input layer
        self.encoder = nn.ModuleList()
        for i in range(0, len(self.layer_features)-1):
            self.encoder.append(nn.Linear(self.layer_features[i], self.layer_features[i+1]))
            if i == len(self.layer_features)-2: ## stop activation, dropout after inserting code layer
                break
            self.encoder.append(layer_activation)
            if dropout > 0.0:
                self.encoder.append(nn.Dropout(p=self.dropout))
                
            
        self.output_layer = nn.Linear(self.code_features, 1)

        ## initialize weights
        self.initialize_weights()

        ## unroll into sequential modules to avoid specifying ModuleList method in forward pass
        self.encoder = nn.Sequential(*self.encoder)

    def forward_encoder(self, x):
        z = self.encoder(x)
        out = self.output_layer(z)
        return out
        
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

# https://datascience.stackexchange.com/questions/75559/why-does-siamese-neural-networks-use-tied-weights-and-how-do-they-work

class SiameseEncoder(nn.Module):

    def __init__(
    self, 
    in_features: int = 2048, 
    code_features: int = 256, 
    layer_activation: nn.Module = nn.ReLU(),
    output_activation: nn.Module = nn.Sigmoid(),
    norm_layer: bool = True,
    masking: Literal['mmp', 'random', 'none'] = 'none',
    similarity_function: Literal['cosine', 'dot_product'] = 'dot_product',
    sparse: bool = False,
    dropout: float = 0.0):

        """
        Encoder for molecular data. Takes ECFP-like features as input and 
        outputs a compressed fingerprint as the autoencoder's latent code.

        Note: currently only works with canonical ECFP sizes e.g. 512, 1024, 2048, 4096
        and corresponding hidden layer combinations

        Args:
            in_features: Number of input features, Uses canonical ECFP Sizes e.g. 512, 1024, 2048, 4096
            code_features: Number of features in the code, i.e. size of the compressed fingerprint
            layer_activation: Activation function to use for the code. 
            output_activation: Activation function to use for the output.
            norm_layer: Whether to use a normalization layer on the code or not. 
            masking: Masking type for the encoder. 
            similarity_function: Similarity function to compute distance in siamese encoder. 
            sparse: Whether to use a sparsity constraint for the code layer or not.
            dropout: Dropout probability. -- Denoising autoencoders
        """

        super().__init__()
        self.in_features = in_features
        self.code_features = code_features

        assert self.in_features % self.code_features == 0, f"Input features must be evenly divisible by code features: {in_features} % {code_features} != 0"
        assert self.in_features > self.code_features, f"Input features must be greater than code features: {in_features} !> {code_features}"
        
        ## calculate layer features from in and code features, halving w.r.t former layer until code layer dimensions are obtained
        ## e.g. 1024, 512, 256, 128, 64, can provide own dimensions as long as it meets assertions below

        self.layer_features = []
        while in_features >= code_features: self.layer_features.append(in_features) ; in_features//=2  
        
        self.norm_layer = norm_layer
        self.dropout = dropout

        self.similarity_function = lambda x, y: torch.dot(x, y) if similarity_function == 'dot_product' else nn.CosineSimilarity()
        assert self.layer_features[-2] > self.code_features, f"Final hidden layer output features must be greater than code features: {self.layer_features[-1]} !> {code_features}"

        self.encoder = nn.ModuleList()

        for i in range(0, len(self.layer_features)-1):
            self.encoder.append(nn.Linear(self.layer_features[i], self.layer_features[i+1]))
            if i == len(self.layer_features)-2: ## stop activation, dropout after inserting code layer
                break
            self.encoder.append(layer_activation)
            if dropout > 0.0:
                self.encoder.append(nn.Dropout(p=self.dropout))

        if self.norm_layer: self.encoder.append(nn.LayerNorm(self.code_features)) 

        self.output_layer = nn.Linear(self.code_features, 1)

        ## initialize weights
        self.initialize_weights()

        ## unroll into sequential modules to avoid specifying ModuleList method in forward pass
        
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z = self.similarity_function(z1, z2)
        out = self.output_layer(z)
        # return self.sigmoid(out) - assumes BCEwithlogits loss will be used
        return out
    
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

class SiameseEncoder(nn.Module):

    def __init__(
    self, 
    in_features: int = 2048, 
    code_features: int = 256, 
    layer_activation: nn.Module = nn.ReLU(),
    output_activation: nn.Module = nn.Sigmoid(),
    norm_layer: bool = True,
    masking: Literal['mmp', 'random', 'none'] = 'none',
    similarity_function: Literal['cosine', 'dot_product'] = 'dot_product',
    sparse: bool = False,
    dropout: float = 0.0):

        """
        Encoder for molecular data. Takes ECFP-like features as input and 
        outputs a compressed fingerprint as the autoencoder's latent code.

        Note: currently only works with canonical ECFP sizes e.g. 512, 1024, 2048, 4096
        and corresponding hidden layer combinations

        Args:
            in_features: Number of input features, Uses canonical ECFP Sizes e.g. 512, 1024, 2048, 4096
            code_features: Number of features in the code, i.e. size of the compressed fingerprint
            layer_activation: Activation function to use for the code. 
            output_activation: Activation function to use for the output.
            norm_layer: Whether to use a normalization layer on the code or not. 
            masking: Masking type for the encoder. 
            similarity_function: Similarity function to compute distance in siamese encoder. 
            sparse: Whether to use a sparsity constraint for the code layer or not.
            dropout: Dropout probability. -- Denoising autoencoders
        """

        super().__init__()
        self.in_features = in_features
        self.code_features = code_features

        assert self.in_features % self.code_features == 0, f"Input features must be evenly divisible by code features: {in_features} % {code_features} != 0"
        assert self.in_features > self.code_features, f"Input features must be greater than code features: {in_features} !> {code_features}"
        
        ## calculate layer features from in and code features, halving w.r.t former layer until code layer dimensions are obtained
        ## e.g. 1024, 512, 256, 128, 64, can provide own dimensions as long as it meets assertions below

        self.layer_features = []
        while in_features >= code_features: self.layer_features.append(in_features) ; in_features//=2  
        
        self.norm_layer = norm_layer
        self.dropout = dropout

        self.similarity_function = lambda x, y: torch.dot(x, y) if similarity_function == 'dot_product' else nn.CosineSimilarity()
        assert self.layer_features[-2] > self.code_features, f"Final hidden layer output features must be greater than code features: {self.layer_features[-1]} !> {code_features}"

        self.encoder = nn.ModuleList()

        for i in range(0, len(self.layer_features)-1):
            self.encoder.append(nn.Linear(self.layer_features[i], self.layer_features[i+1]))
            if i == len(self.layer_features)-2: ## stop activation, dropout after inserting code layer
                break
            self.encoder.append(layer_activation)
            if dropout > 0.0:
                self.encoder.append(nn.Dropout(p=self.dropout))

        if self.norm_layer: self.encoder.append(nn.LayerNorm(self.code_features)) 

        self.output_layer = nn.Linear(self.code_features, 1)

        ## initialize weights
        self.initialize_weights()

        ## unroll into sequential modules to avoid specifying ModuleList method in forward pass
        
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z = self.similarity_function(z1, z2)
        out = self.output_layer(z)
        # return self.sigmoid(out) - assumes BCEwithlogits loss will be used
        return out
    
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