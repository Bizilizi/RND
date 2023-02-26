import typing as t

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid

import wandb
from src.vae_ft.model.vae import MLPVae


class LogRandomImages(Callback):
    def __init__(self, num_images: int = 15, num_rows: int = 5, log_every: int = 5):
        self.num_images = num_images
        self.num_rows = num_rows
        self.log_every = log_every

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        if trainer.current_epoch % self.log_every != 0:
            return None

        model: t.Union[MLPVae] = trainer.model
        model.eval()

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                generated = model.decoder.generate(self.num_images, model.device)

                if not model.decoder.apply_sigmoid:
                    generated = torch.sigmoid(generated)

                generated = (
                    (make_grid(generated, nrow=self.num_rows).permute(1, 2, 0) * 255)
                    .cpu()
                    .numpy()
                )

                images = wandb.Image(generated, caption="Sampled images")

                wandb.log({"train/latent_space/random_images": images})
