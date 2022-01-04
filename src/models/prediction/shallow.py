__all__ = ["ShallowNN"]

from math import floor

import click
import torch
from torch import nn
from torch.optim import Optimizer
from src import data

from src.data.latent import LatentDataModule
from src.data.snp import SNPDataModule
from src.models.layers import make_2d
from src.models.logger import CSVLogger
from src.models.prediction.base import PredictionModel
from src.models.training import train_model
from src.visualization.metrics import plot_metrics
from src.visualization.projection import generate_projection


class ShallowNN(PredictionModel):
    """Shallow neural network.

    Parameters
    ----------
    num_features : int
        Number of input features
    num_classes : int
        Number of target classes
    latent_features : int, optional
        Number of neurons in the hidden layer, by default 2
    lr : float, optional
        Learning rate, by default 1e-3
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        latent_features: int = 2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, latent_features),
            nn.SiLU(),
            nn.Linear(latent_features, num_classes),
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
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.5)

    @make_2d
    def forward(self, x: torch.Tensor):
        return self.network(x)

    @make_2d
    def project(self, x: torch.Tensor):
        with torch.no_grad():
            return self.network[0](x).numpy()


@click.command()
@click.option("-P", "--num_processes", type=int, default=0, help="Set # of CPUs.")
@click.option(
    "-IM", "--model_name", type=str, default=None, help="Set name of inference model."
)
@click.option(
    "-IV",
    "--model_version",
    type=str,
    default=None,
    help="Set inference model's version.",
)
@click.option(
    "-U", "--num_units", type=int, default=None, help="Set # of units in latent space."
)
@click.option("-V", "--version", default=None, help="Set experiment version.")
def main(num_processes, model_name, model_version, num_units, version) -> None:
    is_latent = False
    # Setup data and model
    if model_name is not None and model_version is not None:
        datamodule = LatentDataModule(
            model_name, model_version, val_size=0.2, num_processes=num_processes
        )
        is_latent = True
    else:
        datamodule = SNPDataModule(val_size=0.2, num_processes=num_processes)
    datamodule.setup(stage="fit")

    # Automatically determine number of hidden units
    if is_latent and num_units is None:
        num_units = floor((datamodule.num_features + datamodule.num_classes) / 2)

    model = ShallowNN(datamodule.num_features, datamodule.num_classes, num_units)

    # Train model
    version = version if not is_latent else f"{model_name}_{model_version}"
    logger = CSVLogger("shallow_nn", version, ["loss", "acc"])
    train_model(model, datamodule, logger, num_processes)

    # Plot metrics
    plot_metrics(logger, (10, 4))

    # Project and plot
    if not is_latent:
        generate_projection(logger, model, data, use_tsne=num_units > 2)


if __name__ == "__main__":
    main()
