from typing import Any

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT, DistributedDataParallel

import wandb

from src.qmae_latent_extension.model.vit_vq_vae import VitVQVae


class LogCodebookHistogram(Callback):
    """
    This callback visualizes code book histograms
    """

    def __init__(
        self,
        log_every: int = 200,
    ):

        self.log_every = log_every

        # codebook probs
        self.feature_avg_probs_outputs = None
        self.feature_avg_probs_outputs_count = 0

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        forward_output = outputs["forward_output"]

        if self.feature_avg_probs_outputs is None:
            self.feature_avg_probs_outputs = forward_output.avg_probs
        else:
            self.feature_avg_probs_outputs += forward_output.avg_probs
        self.feature_avg_probs_outputs_count += 1

    @torch.no_grad()
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        model: VitVQVae = trainer.model
        if isinstance(model, DistributedDataParallel):
            model = model.module

        for logger in trainer.loggers:
            if (
                isinstance(logger, WandbLogger)
                and trainer.current_epoch % self.log_every == 0
            ):
                self.log_avg_probs(
                    model,
                    "train/perplexity_bar",
                    self.feature_avg_probs_outputs
                    / self.feature_avg_probs_outputs_count,
                )

        self.feature_avg_probs_outputs = None
        self.feature_avg_probs_outputs_count = 0

    def log_avg_probs(self, model, name: str, avg_probs: torch.Tensor):
        data = [
            [label, val.cpu().item()]
            for (label, val) in zip(range(avg_probs.shape[0]), avg_probs)
        ]
        table = wandb.Table(data=data, columns=["code_idx", "value"])
        wandb.log(
            {
                f"{name}/experience_step_{model.experience_step}": wandb.plot.bar(
                    table, "code_idx", "value", title="Perplexity bar"
                )
            }
        )
