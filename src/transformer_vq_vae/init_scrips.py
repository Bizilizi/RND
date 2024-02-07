import datetime
import typing as t

import torch
from avalanche.benchmarks import SplitCIFAR10, SplitCIFAR100
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping

from avalanche.evaluation.metrics import timing_metrics
from avalanche.training.plugins import EvaluationPlugin
from torchvision import transforms

from src.rnd.callbacks.log_model import LogModelWightsCallback
from src.transformer_vq_vae.callbacks.codebook_histogram import LogCodebookHistogram
from src.transformer_vq_vae.callbacks.log_dataset import LogDataset
from src.transformer_vq_vae.callbacks.training_reconstions_vis import (
    VisualizeTrainingReconstructions,
)
from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.metrics.vq_vae_confusion_matrix import (
    vq_vae_confusion_matrix_metrics,
)
from src.transformer_vq_vae.metrics.vq_vae_forgetting import vq_vae_forgetting_metrics
from src.transformer_vq_vae.metrics.vq_vae_loss import vq_vae_loss_metrics
from src.transformer_vq_vae.model.vit_vq_vae import VitVQVae
from src.transformer_vq_vae.plugins.mixed_precision_plugin import (
    CustomMixedPrecisionPlugin,
)


def get_benchmark(config: TrainConfig, target_dataset_dir):
    if config.dataset == "cifar10":
        config.image_size = 32
        return SplitCIFAR10(
            n_experiences=config.num_tasks,
            return_task_id=True,
            shuffle=True,
            dataset_root=target_dataset_dir,
            train_transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
                    transforms.RandAugment(),
                ]
            ),
            eval_transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
                ]
            ),
        )
    elif config.dataset == "cifar100":
        config.image_size = 32
        return SplitCIFAR100(
            n_experiences=config.num_tasks,
            return_task_id=True,
            shuffle=True,
            dataset_root=target_dataset_dir,
            train_transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
                ]
            ),
            eval_transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
                ]
            ),
        )


def get_evaluation_plugin(
    benchmark, evaluation_loggers, is_using_wandb
) -> t.Optional[EvaluationPlugin]:
    eval_plugin = EvaluationPlugin(
        timing_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
        ),
        vq_vae_forgetting_metrics(experience=True, stream=True),
        vq_vae_loss_metrics(experience=True, stream=True),
        vq_vae_confusion_matrix_metrics(
            num_classes=benchmark.n_classes,
            class_names=list(map(str, range(benchmark.n_classes))),
            save_image=False,
            stream=True,
            wandb=is_using_wandb,
        ),
        loggers=evaluation_loggers,
    )

    return eval_plugin


def get_model(config: TrainConfig, device: torch.device, benchmark) -> VitVQVae:
    vae = VitVQVae(
        num_class_embeddings=config.num_class_embeddings,
        num_embeddings=config.num_embeddings,
        embedding_dim=config.embedding_dim,
        commitment_cost=config.commitment_cost,
        num_classes=benchmark.n_classes,
        supervised=config.supervised,
        decay=config.decay,
        learning_rate=(
            config.learning_rate
            * config.batch_size
            * config.accumulate_grad_batches
            / 256
        ),
        weight_decay=config.weight_decay,
        mask_ratio=config.mask_ratio,
        mask_token_id=config.num_class_embeddings + config.num_embeddings,
        use_lpips=config.use_lpips,
        precision=config.precision,
        accelerator=config.accelerator,
        batch_size=config.batch_size * config.accumulate_grad_batches,
        num_epochs=config.max_epochs,
        past_cycle_consistency_weight=config.cycle_consistency_loss_weight_for_past,
        current_cycle_consistency_weight=config.cycle_consistency_loss_weight_for_current,
        cycle_consistency_sigma=config.cycle_consistency_sigma,
        past_samples_loss_weight=config.past_samples_loss_weight,
        current_samples_loss_weight=config.current_samples_loss_weight,
        future_samples_loss_weight=config.future_samples_loss_weight,
        quantize_features=config.quantize_features,
        quantize_top_k=config.quantize_top_k,
        separate_codebooks=config.separate_codebooks,
    )
    # vae = torch.compile(vae, mode="reduce-overhead")

    return vae


def get_callbacks(config: TrainConfig) -> t.Callable[[int], t.List[Callback]]:
    return lambda experience_step: [
        #     EarlyStopping(
        #         monitor=f"val/reconstruction_loss/experience_step_{experience_step}",
        #         mode="min",
        #         patience=50,
        #     ),
        LogModelWightsCallback(
            log_every=config.save_model_every,
            checkpoint_path=config.checkpoint_path,
            experience_step=experience_step,
            log_to_wandb=False,
        ),
        LogDataset(),
        VisualizeTrainingReconstructions(log_every=10, name="rec_img_100"),
        VisualizeTrainingReconstructions(
            log_every=100, num_images=1000, w1=10, name="rec_img_1000"
        ),
        LogCodebookHistogram(log_every=50),
    ]


def get_train_plugins(config: TrainConfig):
    plugins = []

    return plugins
