import typing as t

from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT

from avalanche.training.templates import BaseSGDTemplate


class RestoreBestPerformingModel(ModelCheckpoint):
    """
    This callback saves the best performing model at each CL step.
    It later used by NaivePytorchLightning to restore weight before next CL step.
    """

    def __init__(self, monitor: str, *args, **kwargs):
        self._monitor = monitor
        super().__init__(
            dirpath="artifacts/cl_best_model",
            filename="model",
            save_top_k=1,
            monitor=monitor,
            *args,
            **kwargs,
        )

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        experience_step = trainer.model.experience_step

        self.current_score = None
        self.best_k_models = {}
        self.kth_best_model_path = ""
        self.best_model_score = None
        self.best_model_path = ""
        self.last_model_path = ""

        self.monitor = f"{self._monitor}/experience_step_{experience_step}"
