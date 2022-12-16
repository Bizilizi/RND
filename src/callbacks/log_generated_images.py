import typing as t

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer


class LogSampledImagesCallback(Callback):

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: t.Any,
        batch_idx: int,
    ) -> None:
        random_x = outputs["random_x"]

        if random_x:
            self.strategy._after_training_iteration(**self.kwargs)
