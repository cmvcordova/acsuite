import torch

class siamac(nn.Module):
    
    """
    Takes two (pretrained) encoder heads, one for positive and one for negative examples.
    Computes one embedding for each example and then computes the similarity 
    between these embeddings. 
    """
    
    def __init__(
        self,
        positive_head: torch.nn.Module,
        negative_head: torch.nn.Module,
        similarity_function: str = 'dot_product',
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        
        self.positive_head = positive_head
        self.negative_head = negative_head
        self.layer_norm = layer_norm
        
        if similarity_function == 'dot_product':
            self.similarity_function = lambda z1, z2: torch.sum(z1 * z2, dim=1)
        else:
            self.similarity_function = torch.nn.CosineSimilarity(dim=1)
            
        if layer_norm:
            ## adds layer normalization to the output of both encoders
            self.positive_layer_norm = torch.nn.LayerNorm(positive_head.out_features)
            self.negative_layer_norm = torch.nn.LayerNorm(negative_head.out_features)
            
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.positive_head(x1)
        z2 = self.negative_head(x2)
        
        if self.layer_norm:
            z1 = self.positive_layer_norm(z1)
            z2 = self.negative_layer_norm(z2)
            
        z = self.similarity_function(z1, z2)
        return z
    