import umap
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, Subset

import wandb


class VisualizeCodebook(Callback):
    """
    This callback visualizes codebook elements
    """

    def __init__(
        self,
        log_every: int = 50,
    ):
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

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                codebook = model.feature_quantization._embedding.weight.clone()
                codebook = model.projection_head(codebook).cpu()

                if codebook.shape[-1] != 2:
                    codebook = umap.UMAP().fit_transform(codebook)
                    codebook = torch.tensor(codebook)

                wandb.log(
                    {
                        f"train/codebook/experience_step_{experience_step}": wandb.Table(
                            columns=["x", "y"],
                            data=codebook.tolist(),
                        ),
                        "epoch": trainer.current_epoch,
                    }
                )
