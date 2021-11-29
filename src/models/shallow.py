__all__ = ["ShallowNN"]

import click
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn

from src.data import SNPDataModule
from src.models.logger import CSVLogger
from src._typing import Optimizer, Tuple


class ShallowNN(pl.LightningModule):
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
        self.save_hyperparameters("num_units", "lr")

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

    def calculate_loss(self, batch) -> Tuple[torch.Tensor, float]:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y.view(-1))
        y_pred = torch.argmax(logits, dim=1)
        accuracy = (y_pred == y).float().mean().item()
        return loss, accuracy

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, acc = self.calculate_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, acc = self.calculate_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss


@click.command()
@click.option(
    "-U", "--num_units", type=int, default=2, help="Sets # of units in latent space."
)
def main(num_units) -> None:
    data = SNPDataModule(test_size=0.2)
    data.setup(stage="fit")

    model = ShallowNN(data.num_features, data.num_classes, num_units)

    logger = CSVLogger("shallow_nn", ["loss", "acc"])
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss")

    trainer = pl.Trainer(
        logger,
        devices="auto",
        accelerator="auto",
        max_epochs=400,
        callbacks=[early_stopping],
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
