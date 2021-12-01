__all__ = ["LCNetwork"]

import click
import torch
import pytorch_lightning as pl
from torch import nn

from src.data import SNPDataModule
from src.models.layers import LCStack
from src.models.logger import CSVLogger
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


@click.command()
@click.option("-P", "--num_processes", type=int, default=0, help="Set # of CPUs.")
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
def main(num_processes, depth, in_features, out_features, dropout, lr, version) -> None:
    from pytorch_lightning.plugins import DDPPlugin

    data = SNPDataModule(val_size=0.2, num_processes=num_processes)
    data.setup(stage="fit")

    model = LCNetwork(depth, in_features, out_features, data.num_classes, dropout, lr)

    with torch.no_grad():
        dummy = torch.ones(data.batch_size, *data.sample_shape)
        model(dummy)

    logger = CSVLogger("lc_network", version, ["loss", "acc"])
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss")

    trainer = pl.Trainer(
        logger,
        accelerator="cpu",
        num_processes=num_processes,
        max_epochs=400,
        callbacks=[early_stopping],
        plugins=DDPPlugin(find_unused_parameters=False),
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
