import typing as t

import torch
from pytorch_lightning import Callback

from avalanche.evaluation.metrics import timing_metrics
from avalanche.training.plugins import EvaluationPlugin
from src.vq_vae.callbacks.mix_random_samples import MixRandomNoise
from src.vq_vae.configuration.config import TrainConfig
from src.vq_vae.metrics.vq_vae_confusion_matrix import vq_vae_confusion_matrix_metrics
from src.vq_vae.metrics.vq_vae_forgetting import vq_vae_forgetting_metrics
from src.vq_vae.metrics.vq_vae_loss import vq_vae_loss_metrics
from src.vq_vae.model.vq_vae import VQVae


def get_evaluation_plugin(
    benchmark, evaluation_loggers, is_using_wandb
) -> t.Optional[EvaluationPlugin]:
    eval_plugin = EvaluationPlugin(
        timing_metrics(epoch_running=True),
        vq_vae_forgetting_metrics(experience=True, stream=True),
        vq_vae_loss_metrics(experience=True, stream=True),
        vq_vae_confusion_matrix_metrics(
            num_classes=benchmark.n_classes,
            class_names=list(map(str, range(benchmark.n_classes))),
            save_image=False,
            stream=True,
            wandb=is_using_wandb,
        ),
        suppress_warnings=True,
        loggers=evaluation_loggers,
    )

    return eval_plugin


def get_model(config: TrainConfig, device: torch.device) -> VQVae:
    vae = VQVae(
        num_hiddens=config.num_hiddens,
        num_residual_layers=config.num_residual_layers,
        num_residual_hiddens=config.num_residual_hiddens,
        num_embeddings=config.num_embeddings,
        embedding_dim=config.embedding_dim,
        commitment_cost=config.commitment_cost,
        decay=config.decay,
        learning_rate=config.learning_rate,
        regularization_lambda=config.regularization_lambda,
        regularization_dropout=config.regularization_dropout,
        data_variance=0.06328692405746414,
        use_lpips=config.use_lpips,
    )

    return vae


def get_callbacks(config: TrainConfig) -> t.List[Callback]:
    return [MixRandomNoise(num_rand_samples=0, num_rand_noise=config.num_random_noise)]
