import typing as t

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset

import wandb
from src.vae_ft.model.vae import MLPVae


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


class MixRandomImages(Callback):
    """
    This callback generates num_images number of images and adds it to the
    dataset for current CL step
    """

    def __init__(
        self,
        num_rand_samples: int = 5_000,
        num_rand_noise: int = 5_000,
        log_dataset: bool = True,
        num_tasks: int = 5,
    ):
        self.num_rand_samples = num_rand_samples
        self.num_rand_noise = num_rand_noise

        self.original_dataset = None
        self.log_dataset = log_dataset
        self.num_tasks = num_tasks

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        experience_step = trainer.model.experience_step
        model: MLPVae = trainer.model

        rehearsed_data = []
        rehearsed_classes = []

        if experience_step > 0 and self.num_rand_samples != 0:
            generated_samples = self.sample_random_images(model, experience_step)
            rehearsed_data.append(generated_samples)
            rehearsed_classes.extend([-1] * generated_samples.shape[0])

        if self.num_rand_noise != 0 and experience_step < self.num_tasks - 1:
            generated_noise = self.sample_random_noise(experience_step)
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
        generated_samples = []
        num_images_to_generate = self.num_rand_samples * experience_step

        sampling_size = min(200, num_images_to_generate)
        for _ in range(num_images_to_generate // sampling_size):
            samples = model.decoder.generate(sampling_size, model.device)
            if not model.decoder.apply_sigmoid:
                samples = torch.sigmoid(samples)

            generated_samples.append(samples)

        generated_samples = torch.cat(generated_samples).cpu()

        return generated_samples

    def sample_random_noise(self, experience_step: int):
        """
        Samples random noise
        """
        noise = torch.randn(
            self.num_rand_noise * (self.num_tasks - experience_step - 1), 1, 28, 28
        )
        noise = noise - noise.min()
        noise = noise / noise.max()

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

    @staticmethod
    def _rescale_image(image):
        image = torch.clone(image)
        image = (image - image.min()) / image.max() * 255

        return image.int()
