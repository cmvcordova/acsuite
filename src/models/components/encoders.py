import numpy as np
import torch
import torch.nn as nn
from typing import List, Literal, Optional

class HalfStepEncoder(nn.Module):
    """
    Encoder for molecular data. Takes ECFP-like features as input and
    outputs a compressed fingerprint as the autoencoder's latent code.
    HalfStep encoder is a simple encoder that halves the input features 
    at each layer until the code layer is reached.
    
    Currently only works with canonical ECFP sizes e.g. 512, 1024, 2048, 4096
    and corresponding hidden layer combinations. Includes decoder methods
    despite being an encoder, as it is intended to be used as a base class for
    autoencoders as well.
    """
    def __init__(
    self, 
    in_features: int = 2048, 
    code_features: int = 256,  
    layer_activation: nn.Module = nn.ReLU(),
    out_features: Optional[int] = None,
    output_activation: Optional[nn.Module] = None,
    dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.code_features = code_features
        self.out_features = out_features
        self.layer_features = self.calculate_layer_features(in_features, code_features)
        self.encoder = self.create_encoder(self.layer_features, layer_activation, dropout)
        self.output_layer = nn.Identity()

        # Optionally add a final output layer for regression or classification tasks
        if self.out_features is not None:
                output_layer = nn.Linear(self.layer_features[-1], self.out_features)
                if output_activation:
                    output_layer = nn.Sequential(
                        output_layer,
                        output_activation
                    )
                self.output_layer = output_layer

        self.apply(self.initialize_weights)

    def forward(self, x):
        z = self.encoder(x)
        z = self.output_layer(z)
        return z

    @staticmethod
    def calculate_layer_features(in_features: int, code_features: int) -> List[int]:
        """
        Calculate the layer features for each layer in the encoder or decoder 
        halving the features until reaching the latent code layer. 

        e.g. 1024, 512, 256, 128, 64, can provide own dimensions as long as it meets assertions.
        """
        
        assert in_features % code_features == 0, "Input features must be evenly divisible by code features." \
                                                f"Received: {in_features} % {code_features} != 0"
        assert in_features > code_features, "Input features must be greater than code features."\
                                            f"Received{in_features} !> {code_features}"
        layer_features = []
        while in_features >= code_features: 
            layer_features.append(in_features)
            in_features//=2  
        return layer_features
    
    def create_encoder(self, layer_features: List[int], 
                       layer_activation: nn.Module,
                       dropout: float = 0.0,
                       norm_layer: Optional[bool] = False) -> nn.Sequential:
        """
        Creates an encoder from halfstep layers.
        """
        encoder_layers = []
        for i in range(len(layer_features)-1):
            encoder_layers.append(nn.Linear(layer_features[i], layer_features[i+1]))
            if i < len(layer_features)-2:
                encoder_layers.append(layer_activation)            
                if dropout > 0.0:
                    encoder_layers.append(nn.Dropout(p=dropout))
                    
        if norm_layer:
            encoder_layers.append(nn.LayerNorm(layer_features[-1]))
        return nn.Sequential(*encoder_layers)
    
    def create_decoder(self, layer_features: List[int], 
                       layer_activation: nn.Module,
                       output_activation: nn.Module,
                       dropout: float) -> nn.Sequential:
        """
        Creates a decoder from halfstep layers. 
        """
        
        decoder_layers = []
        for i in range(len(layer_features)-1):
            decoder_layers.append(nn.Linear(layer_features[i], layer_features[i+1]))
            if i < len(layer_features)-2:
                decoder_layers.append(layer_activation)
                if dropout > 0.0:
                    decoder_layers.append(nn.Dropout(p=dropout))
                    
        if output_activation is not None:
            decoder_layers.append(output_activation)
        return nn.Sequential(*decoder_layers)
    
    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
class HalfStepAutoEncoder(HalfStepEncoder):
    """
    Autoencoder for molecular data. Takes pairs of ECFP-like features as input,
    and outputs a latent code.
    """    
    def __init__(
    self, 
    in_features: int = 2048, 
    code_features: int = 256,  
    layer_activation: nn.Module = nn.ReLU(),
    output_activation: Optional[nn.Module] = None,
    dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.code_features = code_features
        self.layer_features = self.calculate_layer_features(in_features, code_features)
        self.encoder = self.create_encoder(self.layer_features, layer_activation, dropout)
        # reverse the layer features for the decoder
        self.decoder = self.create_decoder(self.layer_features[::-1], layer_activation, output_activation, dropout)

        self.apply(self.initialize_weights)

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

# https://datascience.stackexchange.com/questions/75559/why-does-siamese-neural-networks-use-tied-weights-and-how-do-they-work
class HalfStepSiameseEncoder(HalfStepEncoder):
    """
    Siamese encoder
    """
    def __init__(
    self, 
    in_features: int = 2048, 
    code_features: int = 256, 
    layer_activation: nn.Module = nn.ReLU(),
    norm_layer: bool = True,
    metric: Literal['cosine', 'euclidean', 'manhattan', "hadamard"] = 'manhattan',
    dropout: float = 0.0,
    out_features: int = 1,
    #masking: Optional[Literal['mmp', 'random']] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.code_features = code_features
        self.set_metric(metric)
        
        self.layer_features = self.calculate_layer_features(in_features, code_features)
        
        self.encoder = self.create_encoder(self.layer_features, 
                                           layer_activation, 
                                           norm_layer, dropout)
        
        if metric == "cosine" or metric == "dot_product":
            self.output_layer = nn.Identity()
        else:
            self.output_layer = nn.Linear(code_features, out_features)

        self.apply(self.initialize_weights)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z = self.metric(z1, z2)
        out = self.output_layer(z)
        return out
    
    def set_metric(self, metric: Literal['cosine', 'euclidean', 'manhattan', "hadamard", "dot_product"]):
        if metric == "hadamard":
            self.metric = self.hadamard
        elif metric == "euclidean":
            self.metric = self.euclidean
        elif metric == "manhattan":
            self.metric = self.manhattan
        elif metric == "cosine": 
            self.metric = nn.CosineSimilarity(dim=1)
        elif metric == "dot_product":
            self.metric = self.dot_product
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
    @staticmethod
    def hadamard(z1, z2):
        return z1 * z2

    @staticmethod
    def euclidean(z1, z2):
        return torch.sqrt(torch.sum((z1 - z2).pow(2), dim=1))

    @staticmethod
    def manhattan(z1, z2):
        return torch.sum(torch.abs(z1 - z2), dim=1)

    @staticmethod
    def dot_product(z1, z2):
        return torch.sum(z1 * z2, dim=1)
class HalfStepSiameseAutoEncoder(HalfStepEncoder):
    """
    Siamese autoencoder
    """
    def __init__(
    self, 
    in_features: int = 2048, 
    code_features: int = 256, 
    layer_activation: nn.Module = nn.ReLU(),
    output_activation: nn.Module = None,
    norm_layer: bool = False,
    dropout: float = 0.0,
    masking: Optional[Literal['mmp', 'random']] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.code_features = code_features

        self.layer_features = self.calculate_layer_features(in_features, code_features)
        
        self.encoder = self.create_encoder(self.layer_features, layer_activation, dropout)
        self.decoder = self.create_decoder(self.layer_features[::-1], layer_activation, output_activation, dropout)

        self.apply(self.initialize_weights)

    def forward_encoder(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        return z1, z2
    
    def forward_decoder(self, z1, z2):
        recon_x1 = self.decoder(z1)
        recon_x2 = self.decoder(z2)
        return recon_x1, recon_x2
    
    def forward(self, x1, x2):
        z1, z2 = self.forward_encoder(x1, x2)
        recon_x1, recon_x2 = self.forward_decoder(z1, z2)
        return recon_x1, recon_x2    
class SimSiam(nn.Module):
    """
    SimSiam 
    """
    def __init__(
    self, 
    in_features: int = 2048, 
    hidden_features:int = 256,
    encoder: nn.Module = None,
    layer_activation: nn.Module = nn.ReLU(), ##todo: pass to encoder
    out_features: int = 2048
    ):
        super().__init__()
        self.encoder = encoder or self.create_simsiam_encoder(in_features, hidden_features, out_features)
        self.predictor = self.create_predictor(out_features, hidden_features, out_features)

        self.apply(self.initialize_weights)

    def forward(self, x1, x2):
        ## head 1: encoding through predictor head
        z1 = self.encoder(x1)
        p1 = self.predictor(z1)
        ## head 2: stopgrad on encoding
        z2 = self.encoder(x2).detach()

        return p1, z2
    
    def create_simsiam_encoder(self, input_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def create_predictor(self, in_features, hidden_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )
        
    def initialize_weights(self, m):
        if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
            if hasattr(m, 'weight'):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.normal_(m.weight, 1.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class HalfStepBarlowTwins(HalfStepEncoder):
    """
    Barlow Twins encoder
    """
    def __init__(
    self, 
    in_features: int = 2048, 
    code_features: int = 256, 
    out_features: int = 1,
    projector_layers: int = 3,
    projector_features: int = 4096,
    layer_activation: nn.Module = nn.ReLU(),
    norm_layer: bool = True,
    similarity_function: Literal['cosine', 'dot_product'] = 'dot_product',
    dropout: float = 0.0,
    batch_norm: bool = True,
    masking: Optional[Literal['mmp', 'random']] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.code_features = code_features
        self.out_features = out_features

        self.layer_features = self.calculate_layer_features(in_features, code_features)
        
        if similarity_function == "dot_product":
            self.similarity_function = lambda z1, z2: torch.sum(z1 * z2, dim=1)
        else:
            self.similarity_function = nn.CosineSimilarity(dim=1)

        self.encoder = self.create_encoder(self.layer_features, 
                                           layer_activation, norm_layer, dropout)
        self.projector = self.create_projector(code_features, 
                                               projector_features, projector_layers)

        self.apply(self.initialize_weights)
    
    def create_projector(self,
                         in_features: int = 256,
                         projection_features: int = 4096,
                         projection_layers: int = 3,):
        
        _projector = []
        for i in range(projection_layers):
            layer_in_features = in_features if i == 0 else projection_features
            layer = nn.Linear(layer_in_features, projection_features, bias=True)
            _projector.append(layer)
            if i < projection_layers - 1:
                _projector.append(nn.BatchNorm1d(projection_features,
                                                 track_running_stats=False))
                _projector.append(nn.ReLU())
        return nn.Sequential(*_projector)

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1,z2
