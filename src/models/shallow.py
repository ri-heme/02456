__all__ = ["ShallowNN"]

import click
from pytorch_lightning import plugins
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn

from src.data import SNPDataModule
from src.models.logger import CSVLogger
from src.models.prediction import PredictionModel
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

    def forward(self, x: torch.Tensor):
        x = x.float().transpose(dim0=-1, dim1=1).flatten(start_dim=1)
        return self.network(x)


@click.command()
@click.option("-P", "--num_processes", type=int, default=0, help="Set # of CPUs.")
@click.option(
    "-U", "--num_units", type=int, default=2, help="Set # of units in latent space."
)
@click.option("-V", "--version", default=None, help="Set experiment version.")
def main(num_processes, num_units, version) -> None:
    from pytorch_lightning.plugins import DDPPlugin

    data = SNPDataModule(val_size=0.2, num_processes=num_processes)
    data.setup(stage="fit")

    model = ShallowNN(data.num_features, data.num_classes, num_units)

    logger = CSVLogger("shallow_nn", version, ["loss", "acc"])
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
