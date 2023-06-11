import datetime
import typing as t

import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping

from avalanche.evaluation.metrics import timing_metrics
from avalanche.training.plugins import EvaluationPlugin
from src.rnd.callbacks.log_model import LogModelWightsCallback
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


def get_model(config: TrainConfig, device: torch.device) -> VitVQVae:
    vae = VitVQVae(
        num_class_embeddings=config.num_class_embeddings,
        num_embeddings=config.num_embeddings,
        embedding_dim=config.embedding_dim,
        commitment_cost=config.commitment_cost,
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
        current_samples_loss_weight=config.current_samples_loss_weight,
        batch_size=config.batch_size * config.accumulate_grad_batches,
        num_epochs=config.max_epochs,
        cycle_consistency_weight=config.cycle_consistency_loss_weight,
        cycle_consistency_sigma=config.cycle_consistency_sigma,
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
            log_every=100,
            checkpoint_path=config.checkpoint_path,
            experience_step=experience_step,
            log_to_wandb=False,
        ),
        LogDataset(),
        VisualizeTrainingReconstructions(log_every=10, name="rec_img_100"),
        VisualizeTrainingReconstructions(
            log_every=100, num_images=1000, w1=10, name="rec_img_1000"
        ),
    ]


def get_train_plugins(config: TrainConfig):
    plugins = []

    return plugins
