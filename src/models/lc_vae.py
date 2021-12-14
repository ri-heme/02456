__all__ = ["LCVAE"]

from math import ceil
from typing import Iterable, Tuple

import click
import numpy as np
import torch
import torch.distributions as dist
from scipy import optimize
from torch import nn

from src.data.preprocessing import SNPDataModule
from src.models.extraction import BaseVAE
from src.models.layers import Block, LCLayer, LCStack, pad
from src.models.logger import CSVLogger
from src.models.training import train_model
from src.visualization.metrics import plot_metrics
from src.visualization.projection import plot_projection


def _find_padding(
    observation_features: int,
    depth: int,
    in_chunk_features: int,
    out_chunk_features: int,
):
    def encode_decode(x: Iterable[float]):
        padding = ceil(x[0])
        # Encode
        out_features = _encode_decode(
            observation_features, padding, depth, in_chunk_features, out_chunk_features
        )
        # Decode
        out_features = _encode_decode(
            out_features, 0, depth, out_chunk_features, in_chunk_features
        )
        # Compare features of reconstruction to observation features (incl. padding)
        return out_features - (observation_features + padding)

    root = optimize.excitingmixing(encode_decode, [1])

    return ceil(root[0])


def _encode_decode(
    observation_features: int,
    padding: int,
    depth: int,
    in_chunk_features: int,
    out_chunk_features: int,
):
    out_features = observation_features + padding
    for _ in range(depth):
        out_features = out_chunk_features * ceil(out_features / in_chunk_features)
    return out_features




class LCVAE(BaseVAE):
    """Variational autoencoder with locally-connected linear blocks in its
    encoder and decoder architecture.

    Parameters
    ----------
    observation_features : int
        Number of observation (input) features
    latent_features : int
        Number of features in latent space
    depth : int
        Number of blocks in the stack
    in_chunk_features : int
        Number of input chunk features of each LC layer
    out_chunk_features : int
        Number of output chunk features of each LC layer
    dropout_rate : int
        Probability to dropout units after each block
    lr : float, optional
        Learning rate, by default 1e-4
    beta: float, optional
        Changes the degree of applied learning pressure during training, thus
        encouraging different learnt representations, by default 1
    """

    def __init__(
        self,
        observation_features: int,
        latent_features: int = 2,
        depth: int = 3,
        in_chunk_features: int = 24,
        out_chunk_features: int = 8,
        dropout_rate: float = 0.0,
        lr: float = 1e-4,
        beta: float = 1.0
    ) -> None:
        super().__init__(beta)
        self.observation_features = observation_features
        self.padding = _find_padding(
            observation_features, depth, in_chunk_features, out_chunk_features
        )
        out_features = _encode_decode(
            observation_features, self.padding, depth, in_chunk_features, out_chunk_features
        )
        self.encoder = LCStack(depth, in_chunk_features, out_chunk_features)
        self.latent = nn.ModuleList([nn.LazyLinear(latent_features) for _ in range(2)])
        self.decoder = nn.Sequential(
            Block(nn.Linear, latent_features, out_features, dropout_rate),
            LCStack(depth - 1, out_chunk_features, in_chunk_features, dropout_rate),
            LCLayer(out_chunk_features, in_chunk_features),
        )
        self.lr = lr
        self.save_hyperparameters()

    @pad
    def forward(self, x: torch.Tensor) -> Tuple[dist.Normal, dist.Normal, dist.Normal, torch.Tensor]:
        return super().forward(x)

    @pad
    def calculate_elbo(self, x: torch.Tensor) -> torch.Tensor:
        return super().calculate_elbo(x)

    @pad
    def project(self, x: torch.Tensor) -> np.ndarray:
        return super().project(x)


@click.command()
@click.option("-P", "--num_processes", type=int, default=0, help="Set # of CPUs.")
@click.option(
    "-L",
    "--latent_features",
    type=int,
    default=2,
    help="Set # of latent features.",
)
@click.option("-N", "--depth", type=int, default=3, help="Set # of blocks in network.")
@click.option(
    "-I",
    "--in_features",
    type=int,
    default=3,
    help="Set # of input features per chunk per block.",
)
@click.option(
    "-O",
    "--out_features",
    type=int,
    default=3,
    help="Set # of output features per chunk per block.",
)
@click.option("-D", "--dropout", type=float, default=0.0, help="Set dropout rate.")
@click.option("--lr", type=float, default=1e-4, help="Set learning rate.")
@click.option("-V", "--version", default=None, help="Set experiment version.")
def main(
    num_processes,
    latent_features,
    depth,
    in_features,
    out_features,
    dropout,
    lr,
    version,
) -> None:
    # Setup data and model
    data = SNPDataModule(num_processes=num_processes)
    data.setup(stage="fit")

    model = LCVAE(
        data.num_features,
        latent_features,
        depth,
        in_features,
        out_features,
        dropout,
        lr,
    )

    # Train model
    logger = CSVLogger("lc_vae", version)
    train_model(model, data, logger, num_processes, model_is_lazy=True)

    # Plot metrics
    plot_metrics(logger, (5, 4))

    # Project and plot
    x, y = next(iter(data.predict_dataloader()))
    z = model.project(x)
    plot_projection(logger, y, z, data.idx_to_class)


if __name__ == "__main__":
    main()
