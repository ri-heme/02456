__all__ = ["PCA"]


import torch
from torch import nn

from src.models.layers import make_2d


class PCA:
    def __init__(self, num_components=2):
        super().__init__()
        self.num_components = num_components

    @make_2d
    def __call__(self, x):
        _, _, V = torch.pca_lowrank(x)
        return torch.matmul(x, V[:, : self.num_components])
