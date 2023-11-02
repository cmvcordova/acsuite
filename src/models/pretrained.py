"""
## Placeholder for pretrained encoders
## https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html

#from models.acamodule import ACAModule
#from .models.components.autoencoder import AutoeEncoder
#from .models.components.siameseae import SiameseAutoEncoder
#from .models.components.sharedsae import SharedSiameseAutoEncoder

import os
from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np


path = './data/models/autoencoder/ae_test.ckpt'

class PretrainedAE(ACAModule):
    ## add model checkpoints, directory to hold them, etc.
    def __init__(self, 
                path_to_model_ckpt = './data/models/autoencoder',
                ckpt_name = None):

        super().__init__()

        self.model_ckpt = path_to_model_ckpt + '/' + ckpt_name
    
    def load_model(self):
        return ACAModule.load_from_checkpoint(self.model_ckpt)
"""