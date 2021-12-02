__all__ = ["BaseVAE"]


import numpy as np
import torch
import torch.distributions as dist
import pytorch_lightning as pl

from src._typing import Tuple, Optimizer
from src.models.layers import make_2d


class BaseVAE(pl.LightningModule):
    def encode(self, x) -> Tuple[torch.Tensor, ...]:
        """Encode the observation into the parameters of the posterior
        distribution.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        mu : torch.Tensor
        log_sigma : torch.Tensor
        """
        x = self.encoder(x)
        mu, log_sigma = [layer(x) for layer in self.latent]
        return mu, log_sigma

    def reparametrize(self, mu, log_sigma) -> Tuple[dist.Normal, torch.Tensor]:
        """Define and sample the posterior using the reparametrization trick.

        Parameters
        ----------
        mu : torch.Tensor
        log_sigma : torch.Tensor

        Returns
        -------
        qz : torch.distributions.Normal
        z : torch.Tensor
        """
        # Define & sample posterior
        qz = dist.Normal(mu, log_sigma.exp())
        z = qz.rsample()
        return qz, z

    def decode(self, z) -> dist.Normal:
        """Decode the latent sample into the features of the observation model.

        Parameters
        ----------
        z : torch.Tensor

        Returns
        -------
        px : dist.Normal
        """
        x_hat = self.decoder(z)
        px = dist.Normal(x_hat, torch.ones(1))
        return px

    @make_2d
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[dist.Normal, dist.Normal, dist.Normal, torch.Tensor]:
        mu, log_sigma = self.encode(x)
        qz, z = self.reparametrize(mu, log_sigma)
        px = self.decode(z)
        pz = dist.Normal(torch.zeros_like(mu), torch.ones_like(log_sigma))
        return px, pz, qz, z

    def calculate_elbo(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the ELBO to evaluate the model's performance.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        elbo : torch.Tensor
        """
        px, pz, qz, z = self(x)
        log_px = px.log_prob(x).sum(-1)
        log_pz = pz.log_prob(z)
        log_qz = qz.log_prob(z)
        kl_divergence = (log_qz - log_pz).sum(-1)
        elbo = -(log_px - kl_divergence).mean()
        return elbo

    def configure_optimizers(self) -> Optimizer:
        """Returns an Adam optimizer with a configurable learning rate.

        Returns
        -------
        torch.optim.Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, _ = batch
        loss = self.calculate_elbo(x)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, _ = batch
        loss = self.calculate_elbo(x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    @make_2d
    def project(self, x: torch.Tensor) -> np.ndarray:
        """Projects input into a low-dimensional representation.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        z : np.ndarray
        """
        with torch.no_grad():
            mu, log_sigma = self.encode(x)
            _, z = self.reparametrize(mu, log_sigma)
            return z.numpy()
