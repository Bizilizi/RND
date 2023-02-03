import typing as t

import torch
from pytorch_lightning import Callback

from avalanche.evaluation.metrics import timing_metrics
from avalanche.training.plugins import EvaluationPlugin
from src.avalanche.callbacks.cl_early_stopping import CLEarlyStopping
from src.vae_ft.callbacks.log_latent_space import LogLatentSpace
from src.vae_ft.callbacks.log_sampled_images import LogRandomImages
from src.vae_ft.configuration.config import TrainConfig
from src.vae_ft.metrics.rnd_forgetting import vae_ft_forgetting_metrics
from src.vae_ft.metrics.vae_loss import vae_ft_loss_metrics
from src.vae_ft.model.vae import MLPVae


def get_evaluation_plugin(evaluation_loggers) -> t.Optional[EvaluationPlugin]:
    eval_plugin = EvaluationPlugin(
        timing_metrics(epoch_running=True),
        vae_ft_forgetting_metrics(experience=True, stream=True),
        vae_ft_loss_metrics(experience=True, stream=True),
        suppress_warnings=True,
        loggers=evaluation_loggers,
    )

    return eval_plugin


def get_model(config: TrainConfig, device: torch.device) -> MLPVae:
    vae = MLPVae(
        z_dim=config.z_dim,
        input_dim=config.input_dim,
    )

    return vae


def get_callbacks() -> t.List[Callback]:
    return [
        LogLatentSpace(num_images=200),
        CLEarlyStopping(monitor="val/loss", mode="min", patience=10),
        LogRandomImages(log_every=1),
    ]
