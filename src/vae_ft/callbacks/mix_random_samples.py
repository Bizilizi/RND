import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset

import wandb
from src.vae_ft.model.vae import MLPVae
import typing as t


class AugmentedDataset(Dataset):
    def __init__(
        self, original_dataset: Dataset, rehearsed_data: torch.Tensor, task_id: int
    ) -> None:
        self.original_dataset = original_dataset
        self.rehearsed_data = rehearsed_data
        self.task_id = task_id

    def __getitem__(self, index):
        if index < len(self.original_dataset):
            return self.original_dataset[index]
        else:
            return (
                self.rehearsed_data[index - len(self.original_dataset)],
                -1,
                self.task_id,
            )

    def __len__(self):
        return len(self.original_dataset) + self.rehearsed_data.shape[0]


class MixRandomImages(Callback):
    """
    This callback generates num_images number of images and adds it to the
    dataset for current CL step
    """

    def __init__(self, num_images: int = 5_000, log_dataset: bool = True):
        self.num_images = num_images
        self.original_dataset = None
        self.log_dataset = log_dataset

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        experience_step = trainer.model.experience_step
        model: MLPVae = trainer.model

        if experience_step > 0:
            if self.original_dataset is None:
                self.original_dataset = trainer.datamodule.train_dataset

            generated_samples = []
            for _ in range(self.num_images // 200):
                samples = model.decoder.generate(self.num_images // 200, model.device)
                generated_samples.append(samples)

            generated_samples = torch.cat(generated_samples).cpu()
            augmented_dataset = AugmentedDataset(
                self.original_dataset, generated_samples, experience_step
            )

            trainer.datamodule.train_dataset = augmented_dataset

            if self.log_dataset:
                columns = [f"col_{i}" for i in range(10)]

                image_data_table = wandb.Table(columns=columns)
                class_data_table = wandb.Table(columns=columns)

                dataset = trainer.datamodule.train_dataset

                for logger in trainer.loggers:
                    if isinstance(logger, WandbLogger):
                        random_idx = torch.randperm(len(dataset))[:500]

                        images = [
                            wandb.Image(
                                self._rescale_image(
                                    dataset[random_idx[i]][0],
                                )
                                .cpu()
                                .numpy(),
                                caption=f"dataset_image_{i}",
                            )
                            for i in range(random_idx.shape[0])
                        ]
                        classes = [
                            dataset[random_idx[i]][1]
                            for i in range(random_idx.shape[0])
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

    @staticmethod
    def _rescale_image(image):
        image = torch.clone(image)
        image = (image - image.min()) / image.max() * 255

        return image.int()
