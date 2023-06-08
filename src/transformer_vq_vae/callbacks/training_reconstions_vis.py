import typing as t

import torch
from einops import rearrange
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid

import wandb
from src.transformer_vq_vae.model.vit_vq_vae import ForwardOutput, VitVQVae


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class VisualizeTrainingReconstructions(Callback):
    """
    This callback visualizes reconstruction for training data
    """

    def __init__(
        self,
        num_images: int = 100,
        w1: int = 2,
        log_every: int = 200,
        name: str = "rec_img",
    ):
        self.image_indices = torch.arange(num_images)
        self.w1 = w1
        self.name = name
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
                mask_images = []
                predicted_images = []
                original_images = []
                for image_indices_chunk in chunks(self.image_indices, 100):
                    images = [
                        dataset[idx][0]["images"][None] for idx in image_indices_chunk
                    ]
                    images = torch.cat(images).to(model.device)

                    forward_data: ForwardOutput = model(images)
                    predicted_val_img = forward_data.x_recon
                    mask = forward_data.mask
                    predicted_val_img = predicted_val_img * mask + images * (1 - mask)

                    mask_images.append((images * (1 - mask)).cpu())
                    predicted_images.append(predicted_val_img.cpu())
                    original_images.append(images.cpu())

                mask_images = torch.cat(mask_images)
                predicted_images = torch.cat(predicted_images)
                original_images = torch.cat(original_images)

                all_images = torch.cat([mask_images, predicted_images, original_images])
                all_images = rearrange(
                    all_images, "(v h1 w1) c h w -> c (w1 v h) (h1 w)", w1=self.w1, v=3
                )

                all_images = self._rescale_image(all_images)

                wandb.log(
                    {
                        f"train/dataset/experience_step_{experience_step}/{self.name}": wandb.Image(
                            all_images.numpy(),
                            caption=f"target_img",
                        )
                    }
                )

    @staticmethod
    def _rescale_image(image):
        image = (image + 0.5) * 255
        image = torch.clamp(image, 0, 255)

        return image.permute(1, 2, 0).int()
