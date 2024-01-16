import typing as t

import math
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class EmbClassifier(pl.LightningModule):
    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        experience_step: int,
        batch_size: int,
        num_epochs: int,
        learning_rate: float = 1e-3,
        dataset_mode: str = "",
    ):
        super().__init__()

        self._dataset_mode = dataset_mode
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._num_epochs = num_epochs

        self.experience_step = experience_step
        self.model = nn.Linear(emb_dim, num_classes)

    def forward(self, z):
        return self.model(z)

    def training_step(self, batch, batch_idx):
        image_emb = batch["embeddings"]
        y = batch["labels"]

        logits = self.forward(image_emb)
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
        image_emb = batch["embeddings"]
        y = batch["labels"]

        logits = self.forward(image_emb)
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
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._learning_rate)

        return optimizer
