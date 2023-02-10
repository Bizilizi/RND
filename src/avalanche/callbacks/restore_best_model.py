import typing as t

from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT

from avalanche.training.templates import BaseSGDTemplate


class RestoreBestPerformingModel(ModelCheckpoint):
    """
    This callback saves the best performing model at each CL step.
    It later used by NaivePytorchLightning to restore weight before next CL step.
    """

    def __init__(self, monitor: str, mode: str, *args, **kwargs):
        self._monitor = monitor
        self.mode = mode

        self.args = (monitor, args)
        self.kwargs = kwargs

        super().__init__(
            dirpath="artifacts/cl_best_model",
            filename="model",
            save_top_k=1,
            monitor=monitor,
            mode=mode,
            verbose=True,
            save_weights_only=True,
            *args,
            **kwargs,
        )

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        experience_step = trainer.model.experience_step

        monitor, args = self.args

        super().__init__(
            dirpath="artifacts/cl_best_model",
            filename="model",
            save_top_k=1,
            monitor=f"{self._monitor}/experience_step_{experience_step}",
            mode=self.mode,
            verbose=True,
            save_weights_only=True,
            *args,
            **self.kwargs,
        )

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        monitor_candidates = self._monitor_candidates(trainer)
        self._save_topk_checkpoint(trainer, monitor_candidates)
