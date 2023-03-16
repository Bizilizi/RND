import typing as t

import torch
from datasets import Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class AugmentedDataset(Dataset):
    def __init__(
        self, original_dataset: Dataset, task_id: int, num_tasks_in_batch: int
    ) -> None:
        self.original_dataset = original_dataset
        self.task_id = task_id
        self.num_tasks_in_batch = num_tasks_in_batch

    def __getitem__(self, index):
        x, y, task_id = self.original_dataset[index]

        return x, y + self.task_id * self.num_tasks_in_batch, task_id

    def __len__(self):
        return len(self.original_dataset)


class Cifar10ClassShift(Callback):
    """
    This callback shifts targets returned from SplitCIFAR10 from avalanche
    At every batch, targets are 0,1 in case of n_experiences=5. However, we want actual id of the class,
    thus we need to shift them w.r.t experience_step
    """

    def __init__(self, num_tasks_in_batch: int):
        super().__init__()

        self.num_tasks_in_batch = num_tasks_in_batch

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        experience_step = trainer.model.experience_step

        augmented_dataset = AugmentedDataset(
            original_dataset=trainer.datamodule.train_dataset,
            task_id=experience_step,
            num_tasks_in_batch=self.num_tasks_in_batch,
        )
        trainer.datamodule.train_dataset = augmented_dataset
