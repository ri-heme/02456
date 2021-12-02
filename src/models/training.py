__all__ = ["train_model"]

import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from src.models.logger import CSVLogger
from src.visualization import plot_metrics
from src._typing import ExperimentVersion


def train_model(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    version: ExperimentVersion = None,
    num_processes: int = 1,
    model_is_lazy: bool = False
) -> CSVLogger:
    # Materialize weights of lazy layers
    if model_is_lazy:
        with torch.no_grad():
            dummy = torch.ones(datamodule.batch_size, *datamodule.sample_shape)
            model(dummy)

    # Set up CSV logger and early stopping
    logger = CSVLogger("shallow_nn", version, ["loss", "acc"])
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss")

    # Train
    trainer = pl.Trainer(
        logger,
        accelerator="cpu",
        num_processes=num_processes,
        max_epochs=400,
        callbacks=[early_stopping],
        plugins=DDPPlugin(find_unused_parameters=False),
    )
    trainer.fit(model, datamodule=datamodule)

    return logger
