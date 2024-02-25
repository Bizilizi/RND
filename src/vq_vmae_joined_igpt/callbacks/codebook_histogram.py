import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import DistributedDataParallel
from torch.utils.data import DataLoader

import wandb
from src.vq_vmae_joined_igpt.model.vq_vmae_joined_igpt import VQVMAEJoinedIgpt


class LogCodebookHistogram(Callback):
    """
    This callback visualizes code book histograms
    """

    def __init__(
        self,
        log_every: int = 200,
        batch_size: int = 128,
    ):

        self.log_every = log_every
        self.batch_size = batch_size

    @torch.no_grad()
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch % self.log_every != 0:
            return

        model = trainer.model
        if isinstance(model, DistributedDataParallel):
            model = model.module

        experience_step = model.experience_step
        dataset = trainer.datamodule.train_dataset

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                )

                labels = []
                class_indices = []
                patch_indices = []

                with torch.no_grad():
                    for x, targets, *_ in dataloader:
                        x = x["images"].to(model.device)
                        y = targets["class"]

                        _, full_features, _ = model.encoder(x)
                        *_, input_ids = model.feature_quantization(full_features)

                        class_input_ids = input_ids.reshape(-1, x.shape[0])[0].cpu()
                        patch_input_ids = input_ids.reshape(-1, x.shape[0])[1:].cpu()

                        class_indices.append(class_input_ids)
                        patch_indices.append(patch_input_ids)
                        labels.append(y)

                class_indices = torch.cat(class_indices)
                patch_indices = torch.cat(patch_indices, dim=1).permute(1, 0)
                labels = torch.cat(labels)

                # log bar plot to wandb
                for label in torch.unique(labels):
                    l_class_indices = torch.unique(
                        class_indices[labels == label], return_counts=True
                    )
                    l_patch_indices = torch.unique(
                        patch_indices[labels == label], return_counts=True
                    )

                    class_data = [
                        [label, val]
                        for (label, val) in zip(l_class_indices[0], l_class_indices[1])
                    ]
                    class_table = wandb.Table(
                        data=class_data, columns=["label", "index"]
                    )

                    patch_data = [
                        [label, val]
                        for (label, val) in zip(l_patch_indices[0], l_patch_indices[1])
                    ]
                    patch_table = wandb.Table(
                        data=patch_data, columns=["label", "index"]
                    )

                    wandb.log(
                        {
                            f"train/codebook_histogram/class_experience_step_{experience_step}": wandb.plot.bar(
                                class_table,
                                "label",
                                "index",
                                title="class codebook histogram",
                            ),
                            f"train/codebook_histogram/patch_experience_step_{experience_step}": wandb.plot.bar(
                                patch_table,
                                "label",
                                "index",
                                title="patch codebook histogram",
                            ),
                        }
                    )
