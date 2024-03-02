import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset

from avalanche.benchmarks import SplitCIFAR10
from src.avalanche.data import PLDataModule
from src.avalanche.strategies import NaivePytorchLightning
from src.vq_vmae_knn.configuration.config import TrainConfig
from src.vq_vmae_knn.data.clf_dataset import ClassificationDataset
from src.vq_vmae_knn.model.classification_head import EmbClassifier
from src.vq_vmae_knn.utils.wrap_empty_indices import (
    wrap_dataset_with_empty_indices,
    WrappedDataset,
)


def train_classifier_on_all_classes(
    strategy: NaivePytorchLightning,
    config: TrainConfig,
    benchmark: SplitCIFAR10,
    device: torch.device,
):
    vq_vae_model = strategy.model.to(device)

    clf_head = EmbClassifier(
        emb_dim=config.embedding_dim,
        num_classes=benchmark.n_classes,
        experience_step=strategy.experience_step,
        dataset_mode="all_cls",
        num_epochs=config.max_epochs_lin_eval,
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
        # callbacks=[
        #     EarlyStopping(
        #         monitor=f"val/all_cls_accuracy/experience_step_{strategy.experience_step}",
        #         mode="max",
        #         patience=100,
        #     )
        # ],
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
):
    vq_vae_model = strategy.model.to(device)

    clf_head = EmbClassifier(
        emb_dim=config.embedding_dim,
        num_classes=benchmark.n_classes,
        experience_step=strategy.experience_step,
        dataset_mode="observed_only_cls",
        num_epochs=config.max_epochs_lin_eval,
        batch_size=128,
    ).to(device)

    train_dataset = ConcatDataset(
        [
            experience.dataset
            for experience in benchmark.train_stream[: strategy.experience_step + 1]
        ]
    )
    train_dataset = ClassificationDataset(
        vq_vae_model=vq_vae_model, dataset=train_dataset
    )

    test_dataset = ConcatDataset(
        [
            experience.dataset
            for experience in benchmark.test_stream[: strategy.experience_step + 1]
        ]
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
        # callbacks=[
        #     EarlyStopping(
        #         monitor=f"val/observed_only_cls_accuracy/experience_step_{strategy.experience_step}",
        #         mode="max",
        #         patience=100,
        #     )
        # ],
        max_epochs=config.max_epochs_lin_eval,
        min_epochs=config.min_epochs_lin_eval,
    )

    trainer.fit(clf_head, datamodule=datamodule)

    return clf_head


@torch.no_grad()
def validate_classifier_on_test_stream(
    strategy: NaivePytorchLightning,
    config: TrainConfig,
    benchmark: SplitCIFAR10,
    device: torch.device,
):
    logger = strategy.train_logger
    vq_vae_model = strategy.model.to(device)
    experience_step = strategy.experience_step

    test_dataset = ConcatDataset(
        [
            experience.dataset
            for experience in benchmark.test_stream[: experience_step + 1]
        ]
    )
    test_dataset = WrappedDataset(test_dataset, config.quantize_top_k, 0)

    datamodule = PLDataModule(
        batch_size=128,
        num_workers=config.num_workers,
        train_dataset=test_dataset,
        val_dataset=test_dataset,
    )

    val_dataloader = datamodule.val_dataloader()
    accuracies = []
    for batch in val_dataloader:
        data, targets, *_ = batch

        targets["class"] = targets["class"].to(device)
        targets["time_tag"] = targets["time_tag"].to(device)

        x = data["images"].to(device)

        forward_output = vq_vae_model.forward(x)
        criterion_output = vq_vae_model.criterion(forward_output, targets)

        accuracies.append(criterion_output.clf_acc)

    accuracy = torch.tensor(accuracies).mean()
    if isinstance(logger, WandbLogger):
        logger.log_metrics(
            {
                "val/classification_accuracy/past_tasks": accuracy.cpu().item(),
                "experience_step": strategy.experience_step,
            }
        )
