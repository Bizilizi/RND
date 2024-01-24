import datetime
import typing as t

import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping

from avalanche.evaluation.metrics import timing_metrics
from avalanche.training.plugins import EvaluationPlugin
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
from src.transformer_vq_vae.model.vit_vq_mae import VQMAE
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


def get_model(config: TrainConfig, device: torch.device, benchmark) -> VQMAE:
    vae = VQMAE(
        # model parameters
        num_embeddings=config.num_embeddings,
        num_embeddings_per_step=config.num_embeddings_per_step,
        embedding_dim=config.embedding_dim,
        commitment_cost=config.commitment_cost,
        mask_token_id=config.num_embeddings,
        decay=config.decay,
        weight_decay=config.weight_decay,
        image_size=config.image_size,
        patch_size=config.patch_size,
        encoder_layer=config.encoder_layer,
        encoder_head=config.encoder_head,
        decoder_layer=config.decoder_layer,
        decoder_head=config.decoder_head,
        mask_ratio=config.mask_ratio,
        use_lpips=config.use_lpips,
        # training coefficients
        num_epochs=config.max_epochs,
        batch_size=config.batch_size * config.accumulate_grad_batches,
        learning_rate=(
            config.learning_rate
            * config.batch_size
            * config.accumulate_grad_batches
            / 256
        ),
        warmup=config.warmup,
        # loss coefficients
        past_cycle_consistency_weight=config.cycle_consistency_loss_weight_for_past,
        current_cycle_consistency_weight=config.cycle_consistency_loss_weight_for_current,
        cycle_consistency_sigma=config.cycle_consistency_sigma,
        past_samples_loss_weight=config.past_samples_loss_weight,
        current_samples_loss_weight=config.current_samples_loss_weight,
        future_samples_loss_weight=config.future_samples_loss_weight,
        data_variance=config.data_variance,
        # quantization params
        quantize_features=config.quantize_features,
        quantize_top_k=config.quantize_top_k,
        perplexity_threshold=config.perplexity_threshold,
        # supervision params
        supervised=config.supervised,
        num_classes=benchmark.n_classes,
    )

    return vae


def get_callbacks(config: TrainConfig) -> t.Callable[[int], t.List[Callback]]:
    dataset_mean = 0 if config.dataset == "tiny-imagenet" else 0.5

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
        LogDataset(mean=dataset_mean),
        VisualizeTrainingReconstructions(
            log_every=100,
            name="rec_img_100",
            mean=dataset_mean,
        ),
        VisualizeTrainingReconstructions(
            log_every=100,
            num_images=1000,
            w1=10,
            name="rec_img_1000",
            mean=dataset_mean,
        ),
        LogCodebookHistogram(log_every=50),
    ]


def get_train_plugins(config: TrainConfig):
    plugins = []

    return plugins
