import typing as t

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from transformers import ImageGPTConfig, ImageGPTForCausalImageModeling

if t.TYPE_CHECKING:
    from src.vq_vae.model.vq_vae import VQVae


class ImageGPTCausal(pl.LightningModule):
    def __init__(
        self, configuration: ImageGPTConfig, vq_vae: "VQVae", experience_step: int
    ):
        super().__init__()

        self.experience_step = experience_step
        self.image_gpt = ImageGPTForCausalImageModeling(configuration)
        self.image_gpt.transformer.wte.weight.data[
            :-1
        ] = vq_vae.vq_vae._embedding.weight.data

        self.__dict__["vq_vae"] = vq_vae

    def forward(self, input_ids):
        return self.image_gpt(input_ids=input_ids)

    def training_step(self, batch, batch_idx):
        output = self.forward(input_ids=batch["input_ids"])
        loss = F.cross_entropy(
            output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
            batch["input_ids"][..., 1:].reshape(-1),
        )

        self.log(
            f"train/image_gpt_loss/experience_step_{self.experience_step}",
            loss,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(input_ids=batch["input_ids"])
        loss = F.cross_entropy(
            output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
            batch["input_ids"][..., 1:].reshape(-1),
        )

        self.log(
            f"val/image_gpt_loss/experience_step_{self.experience_step}",
            loss,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": list(self.image_gpt.parameters())[1:], "lr": 0.001},
                {"params": self.image_gpt.transformer.wte.parameters(), "lr": 0.0001},
            ]
        )
        return optimizer
