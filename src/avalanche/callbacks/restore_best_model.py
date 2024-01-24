import typing as t
from uuid import uuid4

from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT, DistributedDataParallel

from avalanche.training.templates import BaseSGDTemplate


class RestoreBestPerformingModel(ModelCheckpoint):
    """
    This callback saves the best performing model at each CL step.
    It later used by NaivePytorchLightning to restore weight before next CL step.
    """

    def __init__(
        self, path_prefix: str, monitor: str, mode: str, every_n_epochs, *args, **kwargs
    ):
        self._monitor = monitor
        self.mode = mode
        self.path_prefix = path_prefix

        self.args = (monitor, args)
        self.kwargs = kwargs

        super().__init__(
            dirpath=self.path_prefix,
            filename="model",
            save_top_k=1,
            monitor=monitor,
            mode=mode,
            save_weights_only=True,
            every_n_epochs=every_n_epochs,
            *args,
            **kwargs,
        )

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if isinstance(trainer.model, DistributedDataParallel):
            experience_step = trainer.model.module.get_experience_step()
        else:
            experience_step = trainer.model.get_experience_step()

        monitor, args = self.args

        super().__init__(
            dirpath=self.path_prefix,
            filename=f"model_exp_{experience_step}",
            save_top_k=1,
            monitor=f"{self._monitor}/experience_step_{experience_step}",
            mode=self.mode,
            save_weights_only=True,
            *args,
            **self.kwargs,
        )

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        monitor_candidates = self._monitor_candidates(trainer)
        self._save_topk_checkpoint(trainer, monitor_candidates)
