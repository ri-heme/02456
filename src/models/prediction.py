__all__ = ["PredictionModel"]


import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src._typing import Tuple


class PredictionModel(pl.LightningModule):
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
