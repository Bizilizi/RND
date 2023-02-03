import typing as t

from pytorch_lightning import Callback, Trainer

from avalanche.benchmarks import CLExperience
from avalanche.training import Naive
from avalanche.training.templates.base import ExpSequence
from pytorch_lightning.callbacks import ModelCheckpoint

from src.avalanche.callbacks.lightning_training_to_avalanche import (
    PLTrainLoopToAvalancheTrainLoopCallback,
)
from src.avalanche.callbacks.restore_best_model import RestoreBestPerformingModel
from src.rnd.callbacks.log_generated_images import LogSampledImagesCallback

if t.TYPE_CHECKING:
    from pytorch_lightning.loggers import Logger

from src.avalanche.data.default_pl_module import PLDataModule


class NaivePytorchLightning(Naive):
    """
    Wrapper over the Naive strategy from Avalanche framework but
    uses trainer from the Pytorch Lightning instead.
    """

    experience_step: int

    def __init__(
        self,
        train_logger: t.Optional["Logger"],
        max_epochs: int,
        min_epochs: int,
        restore_best_model: bool = False,
        train_mb_num_workers: int = 2,
        resume_from: t.Optional[str] = None,
        accelerator: str = "cpu",
        devices: str = "0,",
        validate_every_n: int = 1,
        accumulate_grad_batches: t.Optional[int] = None,
        callbacks: t.Optional[t.Union[t.List[Callback], Callback]] = None,
        *args,
        **kwargs
    ) -> None:
        self.resume_from = resume_from
        self.train_logger = train_logger
        self.train_mb_num_workers = train_mb_num_workers
        self.accumulate_grad_batches = accumulate_grad_batches
        self.validate_every_n = validate_every_n
        self.accelerator = accelerator
        self.devices = devices
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.restore_best_model = restore_best_model

        # Modify callback to
        self.callbacks = callbacks
        self.callbacks.append(
            PLTrainLoopToAvalancheTrainLoopCallback(strategy=self, **kwargs)
        )
        if self.restore_best_model:
            self.callbacks.append(
                RestoreBestPerformingModel(
                    monitor="val/loss",
                )
            )

        super().__init__(*args, **kwargs)

        self.experience_step = 0

    def _train_exp(
        self,
        experiences: t.Union[CLExperience, ExpSequence],
        eval_streams: t.Optional[t.Sequence[t.Union[CLExperience, ExpSequence]]] = None,
        **kwargs
    ) -> None:
        self.model.experience_step = self.experience_step
        self.model.experience = self.experience

        # Create DataModule
        datamodule = PLDataModule(
            batch_size=self.train_mb_size,
            num_workers=self.train_mb_num_workers,
            train_dataset=experiences.dataset,
            val_dataset=eval_streams[0].dataset if eval_streams else None,
        )

        # Training
        trainer = Trainer(
            check_val_every_n_epoch=self.validate_every_n,
            accelerator=self.accelerator,
            devices=self.devices,
            logger=self.train_logger,
            log_every_n_steps=1,
            max_epochs=self.max_epochs,
            min_epochs=self.min_epochs,
            callbacks=self.callbacks,
            accumulate_grad_batches=self.accumulate_grad_batches,
            resume_from_checkpoint="artifacts/cl_best_model/model"
            if self.restore_best_model and self.experience_step > 0
            else None,
        )

        trainer.fit(self.model, datamodule=datamodule, ckpt_path=self.resume_from)

        self.experience_step += 1
