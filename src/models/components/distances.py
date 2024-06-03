import torch

def hadamard(z1, z2):
    return z1 * z2

def euclidean(z1, z2):
    return (z1 - z2).pow(2).sqrt()

def manhattan(z1, z2):
    return torch.abs(z1 - z2)

def dot_product(z1, z2):
    return torch.sum(z1 * z2, dim=1)