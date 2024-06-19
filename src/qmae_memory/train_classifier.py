import torch
from pytorch_lightning import Trainer
from torch.utils.data import ConcatDataset

from avalanche.benchmarks import SplitCIFAR10
from src.avalanche.data import PLDataModule
from src.avalanche.strategies import NaivePytorchLightning
from src.qmae_memory.configuration.config import TrainConfig
from src.qmae_memory.data.clf_dataset import ClassificationDataset
from src.qmae_memory.model.misc.classification_head import EmbClassifier
from src.qmae_memory.model.misc.resnet import ResNet
from src.qmae_memory.utils.wrap_empty_indices import wrap_dataset


def train_classifier_on_random_memory(
    random_memory,
    strategy: NaivePytorchLightning,
    benchmark: SplitCIFAR10,
    config: TrainConfig,
):
    device = strategy.model.device
    vq_vae_model = strategy.model

    clf_head = EmbClassifier(
        emb_dim=config.enc_embedding_dim,
        num_classes=benchmark.n_classes,
        experience_step=strategy.experience_step,
        dataset_mode="gdumb",
        num_epochs=config.classifier_max_epochs,
        batch_size=128,
    ).to(device)

    train_dataset = ClassificationDataset(
        vq_vae_model=vq_vae_model, dataset=random_memory
    )
    test_dataset = ClassificationDataset(
        vq_vae_model=vq_vae_model,
        dataset=ConcatDataset(
            [experience.dataset for experience in benchmark.test_stream]
        ),
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
        max_epochs=config.classifier_max_epochs,
        min_epochs=config.classifier_min_epochs,
    )

    trainer.fit(clf_head, datamodule=datamodule)

    return clf_head


def train_classifier_on_all_classes(
    strategy: NaivePytorchLightning,
    config: TrainConfig,
    benchmark: SplitCIFAR10,
    device: torch.device,
):
    vq_vae_model = strategy.model.to(device)

    clf_head = EmbClassifier(
        emb_dim=config.img_embedding_dim,
        num_classes=benchmark.n_classes,
        experience_step=strategy.experience_step,
        dataset_mode="all_cls",
        num_epochs=config.classifier_max_epochs,
        batch_size=128,
    ).to(device)

    train_dataset = ConcatDataset(
        [experience.dataset for experience in benchmark.train_stream]
    )
    train_dataset = ClassificationDataset(
        vq_vae_model=vq_vae_model, dataset=train_dataset
    )

    test_dataset = ConcatDataset(
        [experience.dataset for experience in benchmark.test_stream]
    )
    test_dataset = ClassificationDataset(
        vq_vae_model=vq_vae_model, dataset=test_dataset
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
        max_epochs=config.classifier_max_epochs,
        min_epochs=config.classifier_min_epochs,
    )

    trainer.fit(clf_head, datamodule=datamodule)

    return clf_head


def train_classifier_on_observed_only_classes(
    strategy: NaivePytorchLightning,
    config: TrainConfig,
    benchmark: SplitCIFAR10,
    bootstrapped_dataset: ClassificationDataset,
    device: torch.device,
):
    clf_model = ResNet(
        num_classes=benchmark.n_classes,
        experience_step=strategy.experience_step,
        batch_size=128,
        num_epochs=config.classifier_max_epochs,
    ).to(device)

    val_dataset = ConcatDataset(
        [
            wrap_dataset(
                experience.dataset,
                img_embedding_dim=config.img_embedding_dim,
                is_past_domain=True,
            )
            for experience in benchmark.test_stream[: strategy.experience_step + 1]
        ]
    )

    datamodule = PLDataModule(
        batch_size=128,
        num_workers=config.num_workers,
        train_dataset=bootstrapped_dataset,
        val_dataset=val_dataset,
    )

    # Training
    trainer = Trainer(
        check_val_every_n_epoch=strategy.validate_every_n,
        accelerator=strategy.accelerator,
        devices=strategy.devices,
        logger=strategy.train_logger,
        max_epochs=config.classifier_max_epochs,
        min_epochs=config.classifier_min_epochs,
    )

    trainer.fit(clf_model, datamodule=datamodule)

    return clf_model
