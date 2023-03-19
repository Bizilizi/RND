import typing as t

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from src.vq_vae.model.resnet import ResidualStack

if t.TYPE_CHECKING:
    from src.vq_vae.model.vq_vae import VQVae


class CnnClassifier(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        vq_vae: "VQVae",
        experience_step: int,
        learning_rate: float = 1e-3,
        dataset_mode: str = "",
    ):
        super().__init__()

        self._dataset_mode = dataset_mode
        self._learning_rate = learning_rate
        self.experience_step = experience_step
        self.model = nn.Sequential(
            ResidualStack(
                in_channels=in_channels,
                num_hiddens=in_channels,
                num_residual_layers=1,
                num_residual_hiddens=32,
                regularization_dropout=0,
            ),
            nn.Flatten(),
            nn.LazyLinear(num_classes),
        )

        self.__dict__["vq_vae"] = vq_vae

    def forward(self, z):
        return self.model(z)

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch

        with torch.no_grad():
            z = self.vq_vae.encoder(x)
            z = self.vq_vae.pre_vq_conv(z)
            _, quantized, _ = self.vq_vae.vq_vae(z)

        logits = self.forward(z)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean().item()

        self.log(
            f"train/{self._dataset_mode}_loss/experience_step_{self.experience_step}",
            loss,
        )
        self.log(
            f"train/{self._dataset_mode}_accuracy/experience_step_{self.experience_step}",
            acc,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, *_ = batch

        with torch.no_grad():
            z = self.vq_vae.encoder(x)
            z = self.vq_vae.pre_vq_conv(z)
            _, quantized, _ = self.vq_vae.vq_vae(z)

        logits = self.forward(z)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean().item()

        self.log(
            f"val/{self._dataset_mode}_loss/experience_step_{self.experience_step}",
            loss,
        )
        self.log(
            f"val/{self._dataset_mode}_accuracy/experience_step_{self.experience_step}",
            acc,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self._learning_rate,
        )
        return optimizer
