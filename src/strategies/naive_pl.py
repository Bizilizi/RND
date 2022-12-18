import typing as t

from avalanche.benchmarks import CLExperience
from avalanche.training import Naive
from avalanche.training.templates.base import ExpSequence
from pytorch_lightning import Trainer

from src.callbacks.lightning_training_to_avalanche import (
    PLTrainLoopToAvalancheTrainLoopCallback,
)
from src.callbacks.log_generated_images import LogSampledImagesCallback

if t.TYPE_CHECKING:
    from pytorch_lightning.loggers import Logger

from src.data.default_pl_module import PLDataModule


class NaivePytorchLightning(Naive):
    """
    Wrapper over the Naive strategy from Avalanche framework but
    uses trainer from the Pytorch Lightning instead.
    """

    experience_step: int

    def __init__(
        self,
        train_logger: t.Optional["Logger"],
        train_mb_num_workers: int = 2,
        resume_from: t.Optional[str] = None,
        accelerator: str = "cpu",
        devices: str = "0,",
        validate_every_n: int = 1,
        accumulate_grad_batches: t.Optional[int] = None,
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
        super().__init__(*args, **kwargs)

        self.experience_step = 0

    def _train_exp(
        self,
        experiences: t.Union[CLExperience, ExpSequence],
        eval_streams: t.Optional[t.Sequence[t.Union[CLExperience, ExpSequence]]] = None,
        **kwargs
    ) -> None:
        self.model.experience_step = self.experience_step

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
            max_epochs=self.train_epochs,
            callbacks=[
                PLTrainLoopToAvalancheTrainLoopCallback(self, **kwargs),
                LogSampledImagesCallback(strategy=self, num_images=10),
                # LogModelWightsCallback(log_every=self.config.validate_every_n),
            ],
            accumulate_grad_batches=self.accumulate_grad_batches,
        )

        trainer.fit(self.model, datamodule=datamodule, ckpt_path=self.resume_from)

        self.experience_step += 1
