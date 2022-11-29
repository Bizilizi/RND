import typing as t

import pytorch_lightning as pl
import torch
import torch.nn as nn
from avalanche.models import SimpleMLP as AvalancheSimpleMLP


class PLSimpleMLP(AvalancheSimpleMLP, pl.LightningModule):
    def __init__(self, cl_step: int, learning_rate: float = 0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.cl_step = cl_step

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.learning_rate, weight_decay=0.0005
        )

        return optimizer

    def training_step(self, batch: t.Tuple[torch.Tensor, int, int], batch_idx):
        x, y, _ = batch
        prediction = self.forward(x)
        train_loss = self.loss(prediction, y)

        self.log(f"train/loss/cl_step_{self.cl_step}", train_loss)
        return train_loss

    def validation_step(self, batch: t.Tuple[torch.Tensor, int, int], batch_idx):
        x, y, _ = batch
        prediction = self.forward(x)
        val_loss = self.loss(prediction, y)

        self.log(f"val/loss/cl_step_{self.cl_step}", val_loss)
        return val_loss
