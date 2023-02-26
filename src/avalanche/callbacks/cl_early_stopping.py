from pytorch_lightning.callbacks import EarlyStopping


class CLEarlyStopping(EarlyStopping):
    def __init__(self, monitor: str, *args, **kwargs):
        self.args = (monitor, args)
        self.kwargs = kwargs

        super().__init__(monitor, *args, **kwargs)

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        experience_step = trainer.model.experience_step
        monitor, args = self.args

        super().__init__(
            f"{monitor}/experience_step_{experience_step}", *args, **self.kwargs
        )
