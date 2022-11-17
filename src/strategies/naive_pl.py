import typing as t

from avalanche.benchmarks import CLExperience
from avalanche.training import Naive
from avalanche.training.templates.base import ExpSequence
from pytorch_lightning import Trainer

if t.TYPE_CHECKING:
    from pytorch_lightning.loggers import Logger

from src.callbacks.log_model import LogModelWightsCallback
from src.configuration.config import TrainConfig
from src.data.default_pl_module import PLDataModule


class NaivePytorchLightning(Naive):
    """
    Wrapper over the Naive strategy from Avalanche framework but
    uses trainer from the Pytorch Lightning instead.
    """

    def __init__(
        self,
        config: TrainConfig,
        train_logger: t.Optional["Logger"],
        resume_from: t.Optional[str] = None,
        *args,
        **kwargs
    ) -> None:
        self.config = config
        self.resume_from = resume_from
        self.train_logger = train_logger
        super().__init__(*args, **kwargs)

    def train(
        self,
        experiences: t.Union[CLExperience, ExpSequence],
        eval_streams: t.Optional[t.Sequence[t.Union[CLExperience, ExpSequence]]] = None,
        **kwargs
    ) -> None:
        # Create DataModule
        datamodule = PLDataModule(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            train_dataset=experiences.dataset,
        )

        # Training
        trainer = Trainer(
            gpus=self.config.gpus,
            check_val_every_n_epoch=self.config.validate_every_n,
            logger=self.train_logger,
            log_every_n_steps=1,
            max_epochs=self.config.max_epochs,
            callbacks=[
                # LogModelWightsCallback(log_every=self.config.validate_every_n),
            ],
            accumulate_grad_batches=self.config.accumulate_grad_batches,
        )
        trainer.fit(self.model, datamodule=datamodule, ckpt_path=self.resume_from)
