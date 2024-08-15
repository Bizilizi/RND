import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torchvision import models


class ResNet(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        experience_step: int,
        batch_size: int,
        num_epochs: int,
        learning_rate: float = 1e-3,
        dataset_mode: str = "resnet",
    ):
        super().__init__()

        self._dataset_mode = dataset_mode
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._num_epochs = num_epochs

        self.experience_step = experience_step
        self.model = models.resnet18(num_classes=num_classes)

    def forward(self, z):
        return self.model(z)

    def training_step(self, batch, batch_idx):
        data, y, *_ = batch
        images = data["images"]

        logits = self.forward(images)
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
        data, y, *_ = batch
        images = data["images"]

        logits = self.forward(images)
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
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.05,
        )

        return optimizer
