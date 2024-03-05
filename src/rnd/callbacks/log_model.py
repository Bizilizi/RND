import typing as t

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

import wandb


class LogModelWightsCallback(Callback):
    def __init__(
        self,
        local_rank: int,
        log_every=5,
        checkpoint_path: str = "checkpoints",
        model_prefix: str = "model",
        model_description: t.Optional[str] = None,
        log_to_wandb: bool = True,
        experience_step: int = None,
    ):
        super().__init__()
        self.state = {"epochs": 0}
        self.log_every = log_every
        self.checkpoint_path = checkpoint_path
        self.model_prefix = model_prefix
        self.model_description = model_description
        self.log_to_wandb = log_to_wandb
        self.experience_step = experience_step
        self.local_rank = local_rank

    def save_model_weights(self, logger, trainer: "pl.Trainer"):
        if self.experience_step is not None:
            model_ckpt = f"{self.checkpoint_path}/{self.model_prefix}-exp-{self.experience_step}-ep-{self.state['epochs']}.ckpt"
        else:
            model_ckpt = f"{self.checkpoint_path}/{self.model_prefix}-ep-{self.state['epochs']}.ckpt"

        trainer.save_checkpoint(model_ckpt)

    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        model: torch.nn.Module,
        unused=None,
    ):
        self.state["epochs"] += 1

        if self.state["epochs"] % self.log_every == 0:
            for logger in trainer.loggers:
                self.save_model_weights(logger, trainer)

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.state.update(callback_state)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return self.state.copy()
