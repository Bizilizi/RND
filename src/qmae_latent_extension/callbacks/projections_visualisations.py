import umap
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, Subset

import wandb


class VisualizeProjections(Callback):
    """
    This callback visualizes image embeddings in case they have dim = 2
    """

    def __init__(
        self,
        benchmark,
        batch_size=128,
        num_images: int = 100,
        log_every: int = 200,
    ):
        self.benchmark = benchmark
        self.num_images = num_images
        self.batch_size = batch_size
        self.log_every = log_every

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
        dataset_to_project = ConcatDataset(
            [
                experience.dataset
                for experience in self.benchmark.train_stream[: experience_step + 1]
            ]
        )
        random_indices = torch.randperm(len(dataset_to_project))[
            : self.num_images
        ].int()
        dataset_to_project = Subset(dataset_to_project, random_indices)

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                dataloader = DataLoader(
                    dataset_to_project,
                    num_workers=8,
                    batch_size=self.batch_size,
                    shuffle=False,
                )

                image_embs = []
                for x, y, _ in dataloader:
                    x = x.to(model.device)
                    forward_output = model.forward(x)
                    image_embs.append(forward_output.image_emb)

                image_embs = torch.cat(image_embs).cpu()

                if image_embs.shape[-1] != 2:
                    image_embs = umap.UMAP().fit_transform(image_embs)

                image_embs = image_embs.tolist()

                data_table = wandb.Table(columns=["x", "y"], data=image_embs)
                wandb.log(
                    {f"train/projections/experience_step_{experience_step}": data_table}
                )
