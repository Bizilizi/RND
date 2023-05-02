import torch
from avalanche.benchmarks import SplitCIFAR10
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms

from src.avalanche.data import PLDataModule
from src.avalanche.strategies import NaivePytorchLightning
from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.model.classification_head import CnnClassifier
from src.vq_vae.data.clf_dataset import ClassificationDataset
from src.vq_vae.model.image_gpt_casual import ImageGPTCausal


def train_classifier_on_all_classes(
    strategy: NaivePytorchLightning,
    config: TrainConfig,
    benchmark: SplitCIFAR10,
    device: torch.device,
    igpt: ImageGPTCausal,
):
    model = strategy.model.to(device)
    igpt = igpt.to(device)

    clf_head = CnnClassifier(
        num_classes=benchmark.n_classes,
        vq_vae=model,
        experience_step=strategy.experience_step,
        dataset_mode="all_cls",
    ).to(device)

    train_dataset = datasets.CIFAR10(
        root=config.dataset_path,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
            ]
        ),
    )
    train_dataset = ClassificationDataset(
        vq_vae_model=model, igpt=igpt.image_gpt, dataset=train_dataset
    )

    test_dataset = datasets.CIFAR10(
        root=config.dataset_path,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
            ]
        ),
    )
    test_dataset = ClassificationDataset(
        vq_vae_model=model, igpt=igpt.image_gpt, dataset=test_dataset
    )

    datamodule = PLDataModule(
        batch_size=256,
        num_workers=config.num_workers,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )

    # Training
    trainer = Trainer(
        check_val_every_n_epoch=strategy.validate_every_n,
        accelerator=strategy.accelerator,
        devices=strategy.devices,
        logger=strategy.train_logger,
        callbacks=[
            EarlyStopping(
                monitor=f"val/all_cls_accuracy/experience_step_{strategy.experience_step}",
                mode="max",
                patience=50,
            )
        ],
        max_epochs=config.max_epochs_lin_eval,
        min_epochs=config.min_epochs_lin_eval,
    )

    trainer.fit(clf_head, datamodule=datamodule)

    return clf_head


def train_classifier_on_observed_only_classes(
    strategy: NaivePytorchLightning,
    config: TrainConfig,
    benchmark: SplitCIFAR10,
    device: torch.device,
    igpt: ImageGPTCausal,
):
    model = strategy.model.to(device)
    igpt = igpt.to(device)

    clf_head = CnnClassifier(
        in_channels=config.embedding_dim,
        num_classes=benchmark.n_classes,
        vq_vae=model,
        igpt=igpt,
        experience_step=strategy.experience_step,
        dataset_mode="all_cls",
        use_cnn=False,
    ).to(device)

    train_dataset = ConcatDataset(
        [
            experience.dataset
            for experience in benchmark.train_stream[: strategy.experience_step + 1]
        ]
    )
    train_dataset = ClassificationDataset(
        vq_vae_model=model, igpt=igpt, dataset=train_dataset
    )

    test_dataset = ConcatDataset(
        [
            experience.dataset
            for experience in benchmark.test_stream[: strategy.experience_step + 1]
        ]
    )
    test_dataset = ClassificationDataset(
        vq_vae_model=model, igpt=igpt, dataset=test_dataset
    )

    datamodule = PLDataModule(
        batch_size=128,
        num_workers=config.num_workers,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )

    # Training
    trainer = Trainer(
        check_val_every_n_epoch=strategy.validate_every_n,
        accelerator=strategy.accelerator,
        devices=strategy.devices,
        logger=strategy.train_logger,
        callbacks=[
            EarlyStopping(
                monitor=f"val/observed_only_cls_accuracy/experience_step_{strategy.experience_step}",
                mode="max",
                patience=50,
            )
        ],
        max_epochs=config.max_epochs_lin_eval,
        min_epochs=config.min_epochs_lin_eval,
    )

    trainer.fit(clf_head, datamodule=datamodule)

    return clf_head
