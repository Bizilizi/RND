import typing as t

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

import wandb
from pytorch_lightning.utilities.types import DistributedDataParallel


class LogDataset(Callback):
    """
    This callback generates num_images number of images and adds it to the
    dataset for current CL step
    """

    def __init__(self, mean: float):
        self.mean = mean

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if isinstance(trainer.model, DistributedDataParallel):
            experience_step = trainer.model.module.get_experience_step()
        else:
            experience_step = trainer.model.get_experience_step()

        self.log_dataset_table(trainer, experience_step)

    def log_dataset_table(self, trainer: Trainer, experience_step: int) -> None:
        columns = [f"col_{i}" for i in range(10)]

        image_data_table = wandb.Table(columns=columns)
        class_data_table = wandb.Table(columns=columns)

        dataset = trainer.datamodule.train_dataset

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                random_idx = torch.randperm(len(dataset))[:100]

                images = [
                    wandb.Image(
                        self._rescale_image(
                            dataset[random_idx[i]][0]["images"],
                        )
                        .cpu()
                        .numpy(),
                        caption=f"dataset_image_{i}",
                    )
                    for i in range(random_idx.shape[0])
                ]
                classes = [
                    dataset[random_idx[i]][1] for i in range(random_idx.shape[0])
                ]

                for row in self._make_rows(images, num_cols=10):
                    image_data_table.add_data(*row)

                for row in self._make_rows(classes, num_cols=10):
                    class_data_table.add_data(*row)

                wandb.log(
                    {
                        f"train/dataset/experience_step_{experience_step}/images": image_data_table
                    }
                )
                wandb.log(
                    {
                        f"train/dataset/experience_step_{experience_step}/classes": class_data_table
                    }
                )

    @staticmethod
    def _make_rows(
        images: t.List[wandb.Image], num_cols: int
    ) -> t.List[t.List[wandb.Image]]:
        row = []
        for i, image in enumerate(images):
            if i != 0 and i % num_cols == 0:
                yield row
                row = []

            row.append(image)

        yield row

    def _rescale_image(self, image):
        image = torch.clone(image) + self.mean
        image = torch.clamp(image, 0) * 255

        return image.permute(1, 2, 0).int()
