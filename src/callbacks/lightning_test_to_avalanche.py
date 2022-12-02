from avalanche.training.templates import BaseSGDTemplate
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import typing as t


class PLTrainLoopToAvalancheTrainLoopCallback(Callback):
    def __init__(self, strategy: BaseSGDTemplate, **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs

    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: t.Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.strategy._before_eval_iteration(**self.kwargs)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: t.Optional[STEP_OUTPUT],
        batch: t.Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.strategy._after_eval_iteration(**self.kwargs)
