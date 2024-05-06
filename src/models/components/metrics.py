import torch
import math
from torchmetrics import Metric

class CollapseLevel(Metric):
    """
    Metric to compute the collapse level of the output of a self-supervised model.
    The collapse level is defined as 1 - sqrt(d) * std(output), where d is the output dimension.
    
    "Standard deviation of the l2-normalized output" -
    if the outputs collapse to a constant vector, their std over all samples should be 
    0 for each channel.
    X. Chen and K. He, ‘Exploring simple Siamese representation learning’, 2020.
    """
    def __init__(self, out_dim:int = 2048, w:float = 0.9, **kwargs):
        ## careful, implementation has no entry for out_dim
        super().__init__(**kwargs)
        self.add_state('avg_output_std', default = torch.tensor(0.0), dist_reduce_fx='sum')
        self.out_dim = out_dim
        self.w = w
        
    def update(self, output):
        output_std = torch.std(output, 0)
        output_std = output_std.mean()
        
        self.avg_output_std = self.w * self.avg_output_std + (1-self.w) * output_std.item()
        
    def compute(self):
        collapse_level = max(0.0, 1 - math.sqrt(self.out_dim) * self.avg_output_std)
        return collapse_level