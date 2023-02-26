import os
import typing as t
from copy import deepcopy

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from avalanche.benchmarks import CLExperience
from avalanche.training import Naive
from avalanche.training.templates.base import ExpSequence
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
        best_model_path_prefix: str = "",
        train_mb_num_workers: int = 2,
        initial_resume_from: t.Optional[str] = None,
        accelerator: str = "cpu",
        devices: str = "0,",
        validate_every_n: int = 1,
        accumulate_grad_batches: t.Optional[int] = None,
        callbacks: t.Optional[t.Union[t.List[Callback], Callback]] = None,
        *args,
        **kwargs,
    ) -> None:
        self.initial_resume_from = initial_resume_from
        self.train_logger = train_logger
        self.train_mb_num_workers = train_mb_num_workers
        self.accumulate_grad_batches = accumulate_grad_batches
        self.validate_every_n = validate_every_n
        self.accelerator = accelerator
        self.devices = devices
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.best_model_path_prefix = best_model_path_prefix

        # Modify callback to
        self.callbacks = callbacks
        self.callbacks.append(
            PLTrainLoopToAvalancheTrainLoopCallback(strategy=self, **kwargs)
        )

        self.restore_best_model_callback = None
        if self.best_model_path_prefix:
            self.restore_best_model_callback = RestoreBestPerformingModel(
                path_prefix=self.best_model_path_prefix, monitor="val/loss", mode="min"
            )
            self.callbacks.append(self.restore_best_model_callback)

        super().__init__(*args, **kwargs)

        self.experience_step = 0

    def _train_exp(
        self,
        experiences: t.Union[CLExperience, ExpSequence],
        eval_streams: t.Optional[t.Sequence[t.Union[CLExperience, ExpSequence]]] = None,
        **kwargs,
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
            max_epochs=self.max_epochs,
            min_epochs=self.min_epochs,
            callbacks=(
                self.callbacks
                + [
                    EarlyStopping(
                        monitor=f"val/loss/experience_step_{self.experience_step}",
                        mode="min",
                        patience=10,
                    )
                ]
            ),
            accumulate_grad_batches=self.accumulate_grad_batches,
        )

        # Derive from which checkpoint to resume training
        if self.experience_step == 0 and self.initial_resume_from:
            state_dict = torch.load(self.initial_resume_from)["state_dict"]
            self.model.load_state_dict(state_dict)

        trainer.fit(self.model, datamodule=datamodule)

        if self.experience_step > 0 and self.restore_best_model_callback:
            state_dict = torch.load(self.restore_best_model_callback.best_model_path)[
                "state_dict"
            ]
            self.model.load_state_dict(state_dict)
            os.remove(self.restore_best_model_callback.best_model_path)

        self.experience_step += 1
