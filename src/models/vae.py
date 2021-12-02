__all__ = ["VAE"]

import click
import torch
import pytorch_lightning as pl
from torch import nn

from src.data import SNPDataModule
from src.models import logger
from src.models.extraction import BaseVAE
from src.models.layers import LinearStack
from src.models.logger import CSVLogger
from src._typing import List
from src.models.training import train_model
from src.visualization.metrics import plot_metrics
from src.visualization.projection import plot_projection


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


@click.command()
@click.option("-P", "--num_processes", type=int, default=0, help="Set # of CPUs.")
@click.option(
    "-L",
    "--latent_features",
    type=int,
    default=2,
    help="Set # of latent features.",
)
@click.option(
    "-U",
    "--num_units",
    type=int,
    multiple=True,
    default=[256, 128],
    help="Set # of units per linear layer.",
)
@click.option("-D", "--dropout", type=float, default=0.0, help="Set dropout rate.")
@click.option("--lr", type=float, default=1e-4, help="Set learning rate.")
@click.option("-V", "--version", default=None, help="Set experiment version.")
def main(num_processes, latent_features, num_units, dropout, lr, version) -> None:
    # Setup data and model
    data = SNPDataModule(num_processes=num_processes)
    data.setup(stage="fit")

    model = VAE(data.num_features, latent_features, num_units, dropout, lr)

    # Train model
    logger = CSVLogger("vae", version)
    train_model(model, data, logger, num_processes, model_is_lazy=True)

    # Plot metrics
    plot_metrics(logger, (5, 4))

    # Project and plot
    x, y = next(iter(data.predict_dataloader()))
    z = model.project(x)
    plot_projection(logger, y, z, data.idx_to_class)


if __name__ == "__main__":
    main()
