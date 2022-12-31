import typing as t

import torch
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from PIL import Image

if t.TYPE_CHECKING:
    from src.strategies.naive_pl import NaivePytorchLightning


class LogSampledImagesCallback(Callback):
    def __init__(
        self,
        strategy: "NaivePytorchLightning",
        num_images: int,
        sample_randomly: bool = True,
    ):
        self.num_images = num_images
        self.sample_randomly = sample_randomly
        self.strategy = strategy
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
                    caption=f"random_image/ex_{self.strategy.experience_step}/gs_{trainer.global_step}",
                )
                for i in range(torch_images.shape[0])
            ]

            self.data_table.add_data(
                self.strategy.experience_step, trainer.global_step, *artifact_images
            )

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                wandb.log(
                    {
                        f"train/generated_images_{self.strategy.experience_step}": self.data_table
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
        columns = ["experience_step", "train_step"] + [
            f"img_{i}" for i in range(self.num_images)
        ]
        self.data_table = wandb.Table(columns=columns)
