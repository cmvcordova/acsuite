## Placeholder for pretrained encoders
## https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html

from .models.components.autoencoder import AutoeEncoder
from .models.components.siameseae import SiameseAutoEncoder
from .models.components.sharedsae import SharedSiameseAutoEncoder

import os
from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np

class PretrainedAE(AutoEncoder):
    ## add model checkpoints, directory to hold them, etc.
    return NotImplemented