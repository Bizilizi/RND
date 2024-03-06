import os
import typing as t

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.profilers import AdvancedProfiler

from avalanche.benchmarks import CLExperience
from avalanche.training import Naive
from avalanche.training.templates.base import ExpSequence
from torch import distributed

from src.avalanche.callbacks.lightning_training_to_avalanche import (
    PLTrainLoopToAvalancheTrainLoopCallback,
)
from src.avalanche.callbacks.restore_best_model import RestoreBestPerformingModel
from src.rnd.callbacks.log_generated_images import LogSampledImagesCallback
from pytorch_lightning.plugins import PrecisionPlugin

if t.TYPE_CHECKING:
    from pytorch_lightning.loggers import Logger

from src.avalanche.data.default_pl_module import PLDataModule


class NaivePytorchLightning(Naive):
    """
    Wrapper over the Naive strategy from Avalanche framework but
    uses trainer from the Pytorch Lightning instead.
    """

    experience_step: int
    trainer: Trainer

    def __init__(
        self,
        train_logger: t.Optional["Logger"],
        train_plugins: t.List[t.Any],
        max_epochs: t.Union[int, t.List[int]],
        min_epochs: t.Union[int, t.List[int]],
        local_rank: int,
        is_distributed: bool,
        best_model_path_prefix: str = "",
        train_mb_num_workers: int = 2,
        initial_resume_from: t.Optional[str] = None,
        accelerator: str = "cpu",
        devices: str = "0,",
        validate_every_n: int = 1,
        accumulate_grad_batches: t.Optional[int] = None,
        callbacks: t.Optional[t.Callable[[int], t.List[Callback]]] = None,
        precision: str = "32-true",
        *args,
        **kwargs,
    ) -> None:
        self.precision = precision
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
        self.train_plugins = train_plugins

        # Modify callback to
        self.callbacks_factory = callbacks
        self.strategy_callbacks = [
            PLTrainLoopToAvalancheTrainLoopCallback(strategy=self, **kwargs)
        ]

        self.restore_best_model_callback = None
        if self.best_model_path_prefix:
            self.restore_best_model_callback = RestoreBestPerformingModel(
                path_prefix=self.best_model_path_prefix,
                monitor="val/loss",
                mode="min",
                every_n_epochs=self.validate_every_n,
                verbose=False,
            )
            self.strategy_callbacks.append(self.restore_best_model_callback)

        super().__init__(*args, **kwargs)

        self.experience_step = 0

    def _train_exp(
        self,
        experiences: t.Union[CLExperience, ExpSequence],
        eval_streams: t.Optional[t.Sequence[t.Union[CLExperience, ExpSequence]]] = None,
        **kwargs,
    ) -> None:
        self.update_model_experience()
        self.resume_from_checkpoint()

        # Create DataModule
        datamodule = PLDataModule(
            batch_size=self.train_mb_size,
            num_workers=self.train_mb_num_workers,
            train_dataset=experiences.dataset,
            val_dataset=eval_streams[0].dataset if eval_streams else None,
        )

        # Training
        if isinstance(self.max_epochs, list):
            max_epochs = self.max_epochs[self.experience_step]
        else:
            max_epochs = self.max_epochs

        if isinstance(self.max_epochs, list):
            min_epochs = self.min_epochs[self.experience_step]
        else:
            min_epochs = self.min_epochs

        self.trainer = Trainer(
            check_val_every_n_epoch=self.validate_every_n,
            accelerator=self.accelerator,
            devices=self.devices,
            logger=self.train_logger,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            callbacks=(
                self.callbacks_factory(self.experience_step) + self.strategy_callbacks
            ),
            plugins=self.train_plugins,
            precision=self.precision,
            accumulate_grad_batches=self.accumulate_grad_batches,
            log_every_n_steps=2
            # profiler=AdvancedProfiler(
            #     dirpath="/Users/ewriji/Desktop/work/RND/", filename="profiler.logs"
            # ),
        )

        self.trainer.fit(self.model, datamodule=datamodule)
        self.restore_best_model()

    def update_model_experience(self) -> None:
        self.model.experience_step = self.experience_step
        self.model.experience = self.experience

    def resume_from_checkpoint(self) -> None:
        if self.experience_step == 0 and self.initial_resume_from:
            state_dict = torch.load(self.initial_resume_from)["state_dict"]
            self.model.load_state_dict(state_dict)

    def restore_best_model(self) -> None:
        if self.experience_step > 0 and self.restore_best_model_callback is not None:
            list_with_best_model_path = [None]
            if self.local_rank == 0:
                list_with_best_model_path[
                    0
                ] = self.restore_best_model_callback.best_model_path

            distributed.broadcast_object_list(list_with_best_model_path)
            best_model_path = list_with_best_model_path[0]

            state_dict = torch.load(best_model_path)["state_dict"]
            self.model.load_state_dict(state_dict)
