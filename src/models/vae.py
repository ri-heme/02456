__all__ = ["VAE"]


import torch
import pytorch_lightning as pl
import torch.distributions as dist
from torch import nn

from src._typing import Optimizer, Tuple


class VAEBlock(nn.Module):
    def __init__(self, in_features, out_features, activation, dropout_prob=0.0):
        super().__init__()
        layers = [
            nn.Linear(in_features, out_features),
            activation(),
            nn.BatchNorm1d(out_features),
        ]
        if dropout_prob > 0.0:
            layers.append(nn.Dropout(dropout_prob))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VAE(pl.LightningModule):
    def __init__(
        self,
        observation_features,
        latent_features=2,
        activation=nn.SiLU,
        n_units=[256, 128],
    ) -> None:
        super().__init__()
        encoder = []
        for in_features, out_features in zip([observation_features] + n_units, n_units):
            encoder.append(VAEBlock(in_features, out_features, activation, 0.0))
        self.encoder = nn.Sequential(*encoder)
        self.latent = nn.ModuleList(
            [nn.Linear(out_features, latent_features) for _ in range(2)]
        )
        decoder = []
        for in_features, out_features in zip(
            [latent_features] + n_units[::-1], n_units[::-1]
        ):
            decoder.append(VAEBlock(in_features, out_features, activation, 0.0))
        decoder.append(nn.Linear(out_features, observation_features))
        self.decoder = nn.Sequential(*decoder)

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
        h_x = self.encoder(x)
        mu, log_sigma = [layer(h_x) for layer in self.latent]
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

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[dist.Normal, dist.Normal, dist.Normal, torch.Tensor]:
        mu, log_sigma = self.encode(x)
        z, qz = self.reparametrize(mu, log_sigma)
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
        """Returns a stochastic gradient descent optimizer with fixed 0.5
        momentum and a configurable learning rate.

        Returns
        -------
        torch.optim.Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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
