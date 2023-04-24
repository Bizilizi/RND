import typing as t

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset

import wandb
from src.vae_ft.model.vae import MLPVae
from src.vq_vae.train_image_gpt import train_img_gpt_on_observed_only_classes


class AugmentedDataset(Dataset):
    def __init__(
        self,
        original_dataset: Dataset,
        rehearsed_data: torch.Tensor,
        rehearsed_classes: t.List[int],
        task_id: int,
    ) -> None:
        self.original_dataset = original_dataset
        self.rehearsed_data = rehearsed_data
        self.rehearsed_classes = rehearsed_classes
        self.task_id = task_id

    def __getitem__(self, index):
        if index < len(self.original_dataset):
            return self.original_dataset[index]
        else:
            return (
                self.rehearsed_data[index - len(self.original_dataset)],
                self.rehearsed_classes[index - len(self.original_dataset)],
                self.task_id,
            )

    def __len__(self):
        return len(self.original_dataset) + self.rehearsed_data.shape[0]


class MixRandomNoise(Callback):
    """
    This callback generates num_images number of images and adds it to the
    dataset for current CL step
    """

    def __init__(
        self,
        num_rand_noise: int = 5_000,
        log_dataset: bool = True,
        num_tasks: int = 5,
    ):
        self.num_rand_noise = num_rand_noise

        self.original_dataset = None
        self.log_dataset = log_dataset
        self.num_tasks = num_tasks

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        experience_step = trainer.model.experience_step
        model: MLPVae = trainer.model
        data_dim = trainer.datamodule.train_dataset[0][0].shape

        rehearsed_data = []
        rehearsed_classes = []

        if self.num_rand_noise != 0 and experience_step < self.num_tasks - 1:
            generated_noise = self.sample_random_noise(experience_step, data_dim)
            rehearsed_data.append(generated_noise)
            rehearsed_classes.extend([-2] * generated_noise.shape[0])

        if rehearsed_data:
            rehearsed_data = torch.cat(rehearsed_data)

            augmented_dataset = AugmentedDataset(
                original_dataset=trainer.datamodule.train_dataset,
                rehearsed_data=rehearsed_data,
                rehearsed_classes=rehearsed_classes,
                task_id=experience_step,
            )

            trainer.datamodule.train_dataset = augmented_dataset

        if self.log_dataset:
            self.log_dataset_table(trainer, experience_step)

    def sample_random_images(self, model: MLPVae, experience_step: int) -> torch.Tensor:
        """
        Samples random images from the model latent space
        """
        ...

    def sample_random_noise(self, experience_step: int, data_dim):
        """
        Samples random noise
        """
        noise = torch.randn(
            self.num_rand_noise * (self.num_tasks - experience_step - 1), *data_dim
        )

        return noise

    def log_dataset_table(self, trainer: Trainer, experience_step: int) -> None:
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
                    dataset[random_idx[i]][1].item()
                    if isinstance(dataset[random_idx[i]][1], torch.Tensor)
                    else dataset[random_idx[i]][1]
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
        image = torch.clone(image) + 0.5
        image = torch.clamp(image, 0) * 255

        return image.permute(1, 2, 0).int()
