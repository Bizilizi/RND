import typing as t

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid
import wandb

from src.transformer_vq_vae.model.vit_vq_vae import VitVQVae, ForwardOutput


class VisualizeTrainingReconstructions(Callback):
    """
    This callback visualizes reconstruction for training data
    """

    def __init__(
        self,
        num_images: int = 100,
        log_every: int = 200,
    ):
        self.image_indices = torch.arange(num_images)
        self.log_every = log_every

    @torch.no_grad()
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch % self.log_every != 0:
            return

        experience_step = trainer.model.experience_step
        model: VitVQVae = trainer.model
        dataset = trainer.datamodule.train_dataset

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                images = [
                    self._rescale_image(
                        dataset[idx][0],
                    )[None]
                    for idx in self.image_indices
                ]
                images = torch.cat(images).to(model.device)
                forward_data: ForwardOutput = model(images)

                target_img = self._rescale_image(make_grid(images))
                rec_img = self._rescale_image(make_grid(forward_data.x_recon))

                wandb.log(
                    {
                        f"train/dataset/experience_step_{experience_step}/target_img": wandb.Image(
                            target_img.cpu().numpy(),
                            caption=f"target_img",
                        )
                    }
                )
                wandb.log(
                    {
                        f"train/dataset/experience_step_{experience_step}/rec_img": wandb.Image(
                            rec_img.cpu().numpy(),
                            caption=f"rec_img",
                        )
                    }
                )

    @staticmethod
    def _rescale_image(image):
        image = (image + 0.5) * 255
        image = torch.clamp(image, 0, 255)

        return image.permute(1, 2, 0).int()
