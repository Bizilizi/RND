import typing as t

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from avalanche.benchmarks import CLExperience, NCExperience
from src.avalanche.model.cl_model import CLModel
from src.vae_ft.model.decoder.cnn import CNNDecoder
from src.vae_ft.model.decoder.mlp import MLPDecoder
from src.vae_ft.model.encoder.cnn import CNNEncoder
from src.vae_ft.model.encoder.mlp import MLPEncoder


class MLPVae(CLModel):
    def __init__(
        self,
        z_dim: int,
        input_dim: int,
        *,
        learning_rate: float = 0.03,
        backbone: str = "mlp",
        regularization: str = "",
        regularization_dropout: float = 0.5,
        regularization_lambda: float = 0.01,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.z_dim = z_dim
        self.regularization = regularization
        self.regularization_lambda = regularization_lambda

        if backbone == "mlp":
            self.encoder = MLPEncoder(
                output_dim=z_dim,
                input_dim=input_dim,
                regularization=regularization,
                dropout=regularization_dropout,
            )
            self.decoder = MLPDecoder(
                input_dim=input_dim,
                h_dim1=512,
                h_dim2=256,
                z_dim=z_dim,
                apply_sigmoid=True,
                regularization=regularization,
                dropout=regularization_dropout,
            )
        elif backbone == "cnn":
            self.encoder = CNNEncoder(
                input_chanel=1,
                output_dim=z_dim,
                regularization=regularization,
                dropout=regularization_dropout,
            )
            self.decoder = CNNDecoder(
                input_dim=z_dim,
                apply_sigmoid=True,
                regularization=regularization,
                dropout=regularization_dropout,
            )
        else:
            assert False, "VAE. Wrong backbone type!"

    @staticmethod
    def criterion(
        x: t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        y: t.Optional[torch.Tensor] = None,
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        x_pred, x_input, log_sigma, mu = x

        kl_div = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        reconstruction_loss = F.binary_cross_entropy(
            x_pred.flatten(1), x_input.flatten(1), reduction="sum"
        )

        return kl_div, reconstruction_loss

    def forward(self, x):
        mu, log_sigma = self.encoder(x)

        # sampling
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        # compute losses
        x_pred = self.decoder(z)

        return x_pred, x, log_sigma, mu

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch

        x_pred, _, log_sigma, mu = self.forward(x)

        # We don't calculate KL loss for the noise, so we filter it out.
        # Noise inputs has class -2
        non_zero_mask = y != -2
        log_sigma = log_sigma[non_zero_mask]
        mu = mu[non_zero_mask]

        # Calculate final VAE loss
        kl_div, reconstruction_loss = self.criterion((x_pred, x, log_sigma, mu))
        loss = kl_div + reconstruction_loss

        self.log_with_postfix(
            f"train/loss",
            loss,
        )
        self.log_with_postfix(
            f"train/kl_loss",
            kl_div,
        )
        self.log_with_postfix(
            f"train/reconstruction_loss",
            reconstruction_loss,
        )

        return {
            "loss": loss,
            "forward_output": x,
        }

    def validation_step(self, batch, batch_idx):
        x, y, *_ = batch

        x_pred, _, log_sigma, mu = self.forward(x)

        kl_div, reconstruction_loss = self.criterion((x_pred, x, log_sigma, mu))
        loss = kl_div + reconstruction_loss

        self.log(
            f"val/loss",
            loss,
        )
        self.log_with_postfix(
            f"val/loss",
            loss,
        )
        self.log_with_postfix(
            f"val/kl_loss",
            kl_div,
        )
        self.log_with_postfix(
            f"val/reconstruction_loss",
            reconstruction_loss,
        )

        return {
            "loss": loss,
            "forward_output": x,
        }

    def configure_optimizers(self):
        weight_decay = (
            self.regularization_lambda if self.regularization == "weight" else None
        )
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=weight_decay
        )
        return optimizer

    def log_with_postfix(self, name: str, value: t.Any, *args, **kwargs):
        self.log_dict(
            {
                f"{name}/experience_step_{self.experience_step}": value,
            },
            *args,
            **kwargs,
        )
