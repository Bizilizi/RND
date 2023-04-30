import typing as t

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

if t.TYPE_CHECKING:
    from src.vq_vae.model.vq_vae import VQVae
    from src.vq_vae.model.image_gpt_casual import ImageGPTCausal


class CnnClassifier(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        vq_vae: "VQVae",
        igpt: "ImageGPTCausal",
        experience_step: int,
        learning_rate: float = 1e-3,
        dataset_mode: str = "",
        use_cnn: bool = False,
    ):
        super().__init__()

        self._dataset_mode = dataset_mode
        self._learning_rate = learning_rate
        self.experience_step = experience_step

        self.model = nn.LazyLinear(num_classes)

        self.__dict__["vq_vae"] = vq_vae
        self.__dict__["igpt"] = igpt

    def forward(self, z):
        return self.model(z)

    def training_step(self, batch, batch_idx):
        y = batch["labels"]
        image_emb = batch["embeddings"]

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
        y = batch["labels"]
        image_emb = batch["embeddings"]

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
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self._learning_rate,
        )
        return optimizer
