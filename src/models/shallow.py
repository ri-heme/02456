__all__ = ["ShallowNN"]

import click
from pytorch_lightning import plugins
import torch
from torch import nn

from src.data import SNPDataModule
from src.models.layers import make_2d
from src.models.prediction import PredictionModel
from src.models.training import train_model
from src.visualization import plot_metrics
from src._typing import Optimizer


class ShallowNN(PredictionModel):
    """Shallow neural network.

    Parameters
    ----------
    num_features : int
        Number of input features
    num_classes : int
        Number of target classes
    num_units : int, optional
        Number of neurons in the hidden layer, by default 2
    lr : float, optional
        Learning rate, by default 1e-3
    """

    def __init__(
        self, num_features: int, num_classes: int, num_units: int = 2, lr: float = 1e-3
    ):
        super().__init__()
        self.lr = lr
        self.network = nn.Sequential(
            nn.Linear(num_features, num_units),
            nn.SiLU(),
            nn.Linear(num_units, num_classes),
            nn.LogSoftmax(dim=1),
        )
        self.save_hyperparameters()

    def configure_optimizers(self) -> Optimizer:
        """Returns a stochastic gradient descent optimizer with fixed 0.5
        momentum and a configurable learning rate.

        Returns
        -------
        torch.optim.Optimizer
        """
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.5)

    @make_2d
    def forward(self, x: torch.Tensor):
        return self.network(x)


@click.command()
@click.option("-P", "--num_processes", type=int, default=0, help="Set # of CPUs.")
@click.option(
    "-U", "--num_units", type=int, default=2, help="Set # of units in latent space."
)
@click.option("-V", "--version", default=None, help="Set experiment version.")
def main(num_processes, num_units, version) -> None:
    # Setup data and model
    data = SNPDataModule(val_size=0.2, num_processes=num_processes)
    data.setup(stage="fit")

    model = ShallowNN(data.num_features, data.num_classes, num_units)

    # Train model
    logger = train_model(model, data, version, num_processes)

    # Plot metrics
    plot_metrics(logger, (10, 4))


if __name__ == "__main__":
    main()
