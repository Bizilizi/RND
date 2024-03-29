import typing as t

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

import wandb

if t.TYPE_CHECKING:
    from src.avalanche.strategies.naive_pl import NaivePytorchLightning


class LogSampledImagesCallback(Callback):
    def __init__(
        self,
        num_images: int,
        sample_randomly: bool = True,
    ):
        self.num_images = num_images
        self.sample_randomly = sample_randomly
        self.data_table = None

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: t.Any,
        batch_idx: int,
    ) -> None:
        random_x = outputs["random_images"]
        random_images_rnd_loss = outputs["random_images_rnd_loss"]

        if self.data_table is None:
            self._init_table()

        if random_x is not None:
            if self.sample_randomly:
                images_idx = torch.randperm(random_x.shape[0])[: self.num_images]
            else:
                images_idx = torch.arange(0, self.num_images)

            torch_images = random_x[images_idx].squeeze()

            artifact_images = [
                wandb.Image(
                    self._rescale_image(torch_images[i]).cpu().numpy(),
                    caption=f"random_image/ex_{trainer.model.experience_step}/gs_{trainer.global_step}",
                )
                for i in range(torch_images.shape[0])
            ]

            artifact_data = [
                el
                for pair in zip(artifact_images, random_images_rnd_loss)
                for el in pair
            ]
            self.data_table.add_data(artifact_data)

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                wandb.log(
                    {
                        f"train/generated_images/experience_step_{trainer.model.experience_step}": self.data_table
                    }
                )
                self.data_table = None

    @staticmethod
    def _rescale_image(image):
        image = torch.clone(image)
        image = (image - image.min()) / image.max() * 255

        return image.int()

    def _init_table(self):
        # Create table
        image_columns = [(f"img_{i}", f"rnd_{i}") for i in range(self.num_images)]
        image_columns = [el for pair in image_columns for el in pair]

        columns = ["experience_step", "train_step"] + image_columns
        self.data_table = wandb.Table(columns=columns)
