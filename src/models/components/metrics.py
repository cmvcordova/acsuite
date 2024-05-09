from typing import Any, Optional, Tuple
import math
import torch

from torch import Tensor, tensor
from torchmetrics import Metric
import torch.nn.functional as F

class CollapseLevel(Metric):
    """
    Metric to compute the collapse level of the output of a self-supervised model.
    The collapse level is defined as 1 - sqrt(d) * std(output), where d is the output dimension.
    
    Reference:
    X. Chen and K. He, ‘Exploring simple Siamese representation learning’, 2020.
    
    Args:
        w (Optional[float]): Weight for the moving average. Default is 0.9.
        kwargs (Any): Additional keyword arguments for the Metric class.
    """
    def __init__(self, w: Optional[float] = 0.9, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        
        self.add_state('avg_output_std_p1', default=tensor(0.0), dist_reduce_fx='mean')
        self.add_state('avg_output_std_p2', default=tensor(0.0), dist_reduce_fx='mean')
        self.w = w

    def update(self, p1: Tensor, p2: Tensor) -> None:
        """
        Update state with collapse level of both calculated projections.
        
        Args:
            p1 (Tensor): First projection vector.
            p2 (Tensor): Second projection vector.
        """
        output_norm_p1 = F.normalize(p1, dim=1)
        output_norm_p2 = F.normalize(p2, dim=1)
        
        output_std_p1 = torch.std(output_norm_p1, 0).mean()
        output_std_p2 = torch.std(output_norm_p2, 0).mean()
        
        self.avg_output_std_p1 = self.w * self.avg_output_std_p1 + (1-self.w) * output_std_p1.item()
        self.avg_output_std_p2 = self.w * self.avg_output_std_p2 + (1-self.w) * output_std_p2.item()
    
        self.out_dim = p1.size(1)  # assuming p1 and p2 have the same size

    def compute(self) -> Tuple[float, float]:
        """
        Compute the collapse level for both projection vectors.
        
        Returns:
            collapse_level_p1 (float): Collapse level for the first projection vector.
            collapse_level_p2 (float): Collapse level for the second projection vector.
        """
        collapse_level_p1 = max(0.0, 1 - math.sqrt(self.out_dim) * self.avg_output_std_p1)
        collapse_level_p2 = max(0.0, 1 - math.sqrt(self.out_dim) * self.avg_output_std_p2)
        return collapse_level_p1#, collapse_level_p2
