import numpy as np
import torch
from pytorch_lightning.callbacks import EarlyStopping


class CLEarlyStopping(EarlyStopping):
    def __init__(self, monitor: str, *args, **kwargs):
        self._monitor = monitor
        super().__init__(monitor, *args, **kwargs)

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        experience_step = trainer.model.experience_step

        # Reset state variables that was filled from previous step in CL pipeline
        self.wait_count = 0
        self.stopped_epoch = 0

        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf
        self.monitor = f"{self._monitor}/experience_step_{experience_step}"
