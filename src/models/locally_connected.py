__all__ = ["LCNetwork"]


import torch
from torch import nn

from src.models import LCStack
from src.models.prediction import PredictionModel
from src._typing import Optimizer


class LCNetwork(PredictionModel):
    """Neural network composed of locally-connected layers.

    Parameters
    ----------
    depth : int
        Number of LC blocks, each consisting of an LC layer, SiLU activation,
        batch normalization, and dropout
    in_chunk_features : int
        Number of input features per chunk per LC layer
    out_chunk_features : int
        Number of target features per chunk per LC layer
    num_classes : int
        Number of target classes
    dropout_rate : int
        Probability to randomly dropout units after each block, by default 0
    lr : float, optional
        Learning rate, by default 1e-4
    """

    def __init__(
        self,
        depth: int,
        in_chunk_features: int,
        out_chunk_features: int,
        num_classes: int,
        dropout_rate: float = 0.0,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.lr = lr
        self.network = nn.Sequential(
            LCStack(depth, in_chunk_features, out_chunk_features, dropout_rate),
            nn.LazyLinear(num_classes),
            nn.LogSoftmax(dim=1),
        )
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        return self.network(x)

    def configure_optimizers(self) -> Optimizer:
        """Returns an Adam optimizer with configurable learning rate.

        Returns
        -------
        torch.optim.Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
