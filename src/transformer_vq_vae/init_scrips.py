import typing as t

import torch
from pytorch_lightning import Callback

from avalanche.evaluation.metrics import timing_metrics
from avalanche.training.plugins import EvaluationPlugin
from pytorch_lightning.callbacks import EarlyStopping

from src.transformer_vq_vae.callbacks.mix_random_samples import MixRandomNoise
from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.metrics.vq_vae_confusion_matrix import (
    vq_vae_confusion_matrix_metrics,
)
from src.transformer_vq_vae.metrics.vq_vae_forgetting import vq_vae_forgetting_metrics
from src.transformer_vq_vae.metrics.vq_vae_loss import vq_vae_loss_metrics
from src.transformer_vq_vae.model.vit_vq_vae import VitVQVae


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
        num_embeddings=config.num_embeddings,
        embedding_dim=config.embedding_dim,
        commitment_cost=config.commitment_cost,
        decay=config.decay,
        learning_rate=config.learning_rate,
        data_variance=0.06328692405746414,
        embeddings_distance=config.embeddings_distance,
        patch_corruption_rate=config.corruption_rate,
        vq_loss_weight=config.vq_loss_weight,
        reconstruction_loss_weight=config.reconstruction_loss_weight,
        contrastive_loss_loss_weight=config.contrastive_loss_loss_weight,
        encoder_mlm_loss_loss_weight=config.encoder_mlm_loss_loss_weight,
        decoder_regression_loss_loss_weight=config.decoder_regression_loss_loss_weight,
    )
    # vae = torch.compile(vae, mode="reduce-overhead")

    return vae


def get_callbacks(config: TrainConfig) -> t.Callable[[int], t.List[Callback]]:
    return lambda experience_step: [
        EarlyStopping(
            monitor=f"val/reconstruction_loss/experience_step_{experience_step}",
            mode="min",
            patience=50,
        )
    ]
