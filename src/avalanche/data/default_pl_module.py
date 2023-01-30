import typing as t

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class PLDataModule(pl.LightningDataModule):
    train_dataset: Dataset
    val_dataset: Dataset

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_dataset: Dataset,
        val_dataset: t.Optional[Dataset] = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self) -> t.Optional[DataLoader]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> t.Optional[DataLoader]:
        return (
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
            if self.val_dataset
            else None
        )
