import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Dataset

import wandb
from src.vae_ft.model.vae import MLPVae


class TensorDataset(Dataset):
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    def __getitem__(self, index):
        return self.data[index], -1


class MixRandomImages(Callback):
    """
    This callback generates num_images number of images and adds it to the
    dataset for current CL step
    """

    def __init__(self, num_images: int = 5_000):
        self.num_images = num_images
        self.original_dataset = None

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

            generated_samples = torch.cat(generated_samples)
            generated_dataset = TensorDataset(generated_samples)

            trainer.datamodule.train_dataset = self.original_dataset + generated_dataset
