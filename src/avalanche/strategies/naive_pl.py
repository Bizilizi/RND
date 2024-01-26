import os
import typing as t

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.plugins import PrecisionPlugin
from pytorch_lightning.profilers import AdvancedProfiler

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
    trainer: Trainer

    def __init__(
        self,
        train_logger: t.Optional["Logger"],
        train_plugins: t.List[t.Any],
        max_epochs: int,
        min_epochs: int,
        best_model_path_prefix: str = "",
        train_mb_num_workers: int = 2,
        resume_arguments: t.Dict[str, str] = None,
        accelerator: str = "cpu",
        devices: str = "0,",
        strategy: str = "auto",
        validate_every_n: int = 1,
        accumulate_grad_batches: t.Optional[int] = None,
        callbacks: t.Optional[t.Callable[[int], t.List[Callback]]] = None,
        precision: str = "32-true",
        *args,
        **kwargs,
    ) -> None:
        self.precision = precision
        self.resume_arguments = resume_arguments
        self.train_logger = train_logger
        self.train_mb_num_workers = train_mb_num_workers
        self.accumulate_grad_batches = accumulate_grad_batches
        self.validate_every_n = validate_every_n
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.best_model_path_prefix = best_model_path_prefix
        self.train_plugins = train_plugins

        self.callbacks_factory = callbacks
        self.restore_best_model_callback = None

        super().__init__(*args, **kwargs)

        self.experience_step = (
            resume_arguments["experience_step"] if resume_arguments else 0
        )

    def _train_exp(
        self,
        experiences: t.Union[CLExperience, ExpSequence],
        eval_streams: t.Optional[t.Sequence[t.Union[CLExperience, ExpSequence]]] = None,
        **kwargs,
    ) -> None:
        self.update_model_experience()

        # Create DataModule
        datamodule = PLDataModule(
            batch_size=self.train_mb_size,
            num_workers=self.train_mb_num_workers,
            train_dataset=experiences.dataset,
            val_dataset=eval_streams[0].dataset if eval_streams else None,
        )

        callbacks = [PLTrainLoopToAvalancheTrainLoopCallback(strategy=self, **kwargs)]
        if self.best_model_path_prefix:
            self.restore_best_model_callback = RestoreBestPerformingModel(
                path_prefix=self.best_model_path_prefix,
                monitor="val/reconstruction_loss",
                mode="min",
                every_n_epochs=self.validate_every_n,
                verbose=False,
            )
            callbacks = callbacks + [self.restore_best_model_callback]

        # Training
        self.trainer = Trainer(
            check_val_every_n_epoch=self.validate_every_n,
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            logger=self.train_logger,
            max_epochs=self.max_epochs,
            min_epochs=self.min_epochs,
            callbacks=(self.callbacks_factory(self.experience_step) + callbacks),
            plugins=self.train_plugins,
            precision=self.precision,
            accumulate_grad_batches=self.accumulate_grad_batches,
            log_every_n_steps=2,
            # profiler=AdvancedProfiler(
            #     dirpath="/Users/ewriji/Desktop/work/RND/", filename="profiler.logs"
            # ),
        )

        ckpt_path = (
            self.resume_arguments["model_checkpoint_path"]
            if self.resume_arguments
            else None
        )

        self.trainer.fit(self.model, datamodule=datamodule, ckpt_path=ckpt_path)

        self.resume_arguments = None
        """An easy trick to avoid restoring twice when train loop revisited"""
        self.restore_best_model()
        self.experience_step += 1

    def update_model_experience(self) -> None:
        if hasattr(self.model, "update_model_experience"):
            self.model.update_model_experience(self.experience_step, self.experience)
        else:
            self.model.set_experience_step(self.experience_step)
            self.model.set_experience(self.experience)

    def restore_best_model(self) -> None:
        if self.experience_step > 0 and self.restore_best_model_callback:
            state_dict = torch.load(self.restore_best_model_callback.best_model_path)[
                "state_dict"
            ]
            self.model.load_state_dict(state_dict)
            # os.remove(self.restore_best_model_callback.best_model_path)
