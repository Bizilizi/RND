import datetime
import typing as t

import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping

from avalanche.evaluation.metrics import timing_metrics
from avalanche.training.plugins import EvaluationPlugin
from src.rnd.callbacks.log_model import LogModelWightsCallback
from src.vq_vae.callbacks.mix_random_samples import LogDataset
from src.vq_vae.configuration.config import TrainConfig
from src.vq_vae.metrics.vq_vae_confusion_matrix import vq_vae_confusion_matrix_metrics
from src.vq_vae.metrics.vq_vae_forgetting import vq_vae_forgetting_metrics
from src.vq_vae.metrics.vq_vae_loss import vq_vae_loss_metrics
from src.vq_vae.model.vq_vae import VQVae


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


def get_model(config: TrainConfig, device: torch.device) -> t.Union[VQVae]:
    vae = VQVae(
        num_hiddens=config.num_hiddens,
        num_residual_layers=config.num_residual_layers,
        num_residual_hiddens=config.num_residual_hiddens,
        num_embeddings=config.num_embeddings,
        num_classes=10,
        embedding_dim=config.embedding_dim,
        commitment_cost=config.commitment_cost,
        decay=config.decay,
        learning_rate=config.learning_rate,
        data_variance=0.06328692405746414,
        regularization_dropout=config.regularization_dropout,
        regularization_lambda=config.regularization_lambda,
        vq_loss_weight=config.vq_loss_weight,
        reconstruction_loss_weight=config.reconstruction_loss_weight,
        downstream_loss_weight=config.downstream_loss_weight,
        use_lpips=config.use_lpips,
    )

    return vae


def get_callbacks(config: TrainConfig) -> t.Callable[[int], t.List[Callback]]:
    return lambda x: [
        LogDataset(),
        LogModelWightsCallback(
            log_every=10,
            checkpoint_path=config.checkpoint_path,
            model_prefix="vqvae",
        ),
    ]
