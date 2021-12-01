__all__ = ["VAE"]

import torch
from torch import nn

from src.models.extraction import BaseVAE
from src.models.layers import Block, LinearStack
from src._typing import List


class VAE(BaseVAE):
    """Variational autoencoder with linear blocks in its encoder and decoder
    architecture.

    Parameters
    ----------
    observation_features : int
        Number of observation (input) features
    latent_features : int
        Number of features in latent space
    num_units : list of int
        Number of units in each hidden layer
    dropout_rate : int
        Probability to dropout units after each block
    lr : float, optional
        Learning rate, by default 1e-4
    """

    def __init__(
        self,
        observation_features: int,
        latent_features: int = 2,
        num_units: List[int] = [256, 128],
        dropout_rate: float = 0.0,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.encoder = LinearStack(num_units, dropout_rate)
        self.latent = nn.ModuleList([nn.LazyLinear(latent_features) for _ in range(2)])
        self.decoder = nn.Sequential(
            LinearStack(num_units[::-1], dropout_rate),
            nn.LazyLinear(observation_features),
        )
        self.lr = lr

    def calculate_elbo(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(dim0=-1, dim1=1).flatten(start_dim=1)
        return super().calculate_elbo(x)