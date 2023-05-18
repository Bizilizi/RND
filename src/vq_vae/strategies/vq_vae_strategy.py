import typing as t

from pytorch_lightning import Trainer
from torch.utils.data import Dataset

from avalanche.benchmarks import CLExperience
from avalanche.training.templates.base import ExpSequence
from src.avalanche.data.default_pl_module import PLDataModule
from src.avalanche.strategies import NaivePytorchLightning


class VQVaeStrategy(NaivePytorchLightning):
    def _train_exp(
        self,
        experiences: t.Union[CLExperience, ExpSequence],
        eval_streams: t.Optional[t.Sequence[t.Union[CLExperience, ExpSequence]]],
        train_dataset: Dataset,
        **kwargs,
    ) -> None:
        self.update_model_experience()
        self.resume_from_checkpoint()

        # Create DataModule
        datamodule = PLDataModule(
            batch_size=self.train_mb_size,
            num_workers=self.train_mb_num_workers,
            train_dataset=train_dataset,
            val_dataset=eval_streams[0].dataset if eval_streams else None,
        )

        # Training
        self.trainer = Trainer(
            check_val_every_n_epoch=self.validate_every_n,
            accelerator=self.accelerator,
            devices=self.devices,
            logger=self.train_logger,
            max_epochs=self.max_epochs,
            min_epochs=self.min_epochs,
            callbacks=(
                self.callbacks_factory(self.experience_step) + self.strategy_callbacks
            ),
            accumulate_grad_batches=self.accumulate_grad_batches,
            log_every_n_steps=2,
        )

        self.trainer.fit(self.model, datamodule=datamodule)
        self.restore_best_model()
