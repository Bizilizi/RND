import typing as t

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer

from avalanche.training.templates import BaseSGDTemplate


class PLTrainLoopToAvalancheTrainLoopCallback(Callback):
    def __init__(self, strategy: BaseSGDTemplate, **kwargs):
        super().__init__()
        self.strategy = strategy
        self.kwargs = kwargs

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.strategy._before_training_epoch(**self.kwargs)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.strategy._after_training_epoch(**self.kwargs)

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: t.Any,
        batch_idx: int,
    ) -> None:
        self.strategy.mbatch = batch
        self.strategy.loss = 0
        self.strategy._before_training_iteration(**self.kwargs)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: t.Any,
        batch_idx: int,
    ) -> None:
        self.strategy.mb_output = outputs["forward_output"]
        self.strategy._after_training_iteration(**self.kwargs)

    def on_before_backward(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor
    ) -> None:
        self.strategy.loss = loss
        self.strategy._before_backward(**self.kwargs)

    def on_after_backward(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.strategy._after_backward(**self.kwargs)

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: Optimizer,
    ) -> None:
        self.strategy._before_update(**self.kwargs)

    def on_before_zero_grad(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: Optimizer,
    ) -> None:
        self.strategy._after_update(**self.kwargs)
