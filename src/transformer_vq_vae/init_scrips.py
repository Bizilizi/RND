import datetime
import typing as t

import torch
from avalanche.benchmarks import SplitCIFAR10, SplitCIFAR100
from pytorch_lightning import Callback

from avalanche.evaluation.metrics import timing_metrics
from avalanche.training.plugins import EvaluationPlugin
from torchvision import transforms

from src.avalanche.strategies import NaivePytorchLightning
from src.rnd.callbacks.log_model import LogModelWightsCallback
from src.transformer_vq_vae.callbacks.log_dataset import LogDataset
from src.transformer_vq_vae.callbacks.reconstruction_visualization_plugin import (
    ReconstructionVisualizationPlugin,
)
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
from train_utils import get_loggers


def get_epochs_schedule(config: TrainConfig):
    if config.num_epochs_schedule == "fixed":
        return config.max_epochs

    if config.num_epochs_schedule == "schedule":
        schedule = torch.linspace(
            config.max_epochs, config.min_epochs, config.num_tasks
        )
        return schedule.tolist()

    if config.num_epochs_schedule == "warmup":
        schedule = [config.min_epochs] * config.num_tasks
        schedule[0] = config.max_epochs

        return schedule


def get_cl_strategy(
    *,
    config,
    args,
    wandb_params,
    model,
    benchmark,
    device,
    is_using_wandb,
    is_distributed
):
    cl_strategy_logger, eval_plugin_loggers = get_loggers(config, model, wandb_params)
    evaluation_plugin = get_evaluation_plugin(
        benchmark, eval_plugin_loggers, is_using_wandb
    )

    epochs_schedule = get_epochs_schedule(config)
    cl_strategy = NaivePytorchLightning(
        precision=config.precision,
        accelerator=config.accelerator,
        devices=config.devices,
        validate_every_n=config.validate_every_n,
        accumulate_grad_batches=config.accumulate_grad_batches,
        train_logger=cl_strategy_logger,
        initial_resume_from=args.resume_from,
        model=model,
        device=device,
        optimizer=model.configure_optimizers()["optimizer"],
        criterion=model.criterion,
        train_mb_size=config.batch_size,
        train_mb_num_workers=config.num_workers,
        train_epochs=config.max_epochs,
        eval_mb_size=config.batch_size,
        evaluator=evaluation_plugin,
        callbacks=get_callbacks(config, args.local_rank),
        max_epochs=epochs_schedule,
        min_epochs=epochs_schedule,
        best_model_path_prefix=config.best_model_prefix,
        plugins=[ReconstructionVisualizationPlugin(num_tasks_in_batch=2)],
        train_plugins=get_train_plugins(config),
        is_distributed=is_distributed,
        local_rank=args.local_rank,
    )


def get_benchmark(config: TrainConfig, target_dataset_dir):
    if config.dataset == "cifar10":
        return SplitCIFAR10(
            n_experiences=config.num_tasks,
            return_task_id=True,
            shuffle=True,
            dataset_root=target_dataset_dir,
            train_transform=transforms.Compose(
                [
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
    elif config.dataset == "cifar100":
        config.dataset_variance = 0.071
        return SplitCIFAR100(
            n_experiences=config.num_tasks,
            return_task_id=True,
            shuffle=True,
            dataset_root=target_dataset_dir,
            train_transform=transforms.Compose(
                [
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
        past_samples_loss_weight=config.past_samples_loss_weight,
        batch_size=config.batch_size * config.accumulate_grad_batches,
        num_epochs=config.max_epochs,
        cycle_consistency_weight=config.cycle_consistency_loss_weight,
        cycle_consistency_sigma=config.cycle_consistency_sigma,
        quantize_features=config.quantize_features,
        data_variance=config.dataset_variance,
    )
    # vae = torch.compile(vae, mode="reduce-overhead")

    return vae


def get_callbacks(
    config: TrainConfig, local_rank: int
) -> t.Callable[[int], t.List[Callback]]:
    return lambda experience_step: [
        #     EarlyStopping(
        #         monitor=f"val/reconstruction_loss/experience_step_{experience_step}",
        #         mode="min",
        #         patience=50,
        #     ),
        LogModelWightsCallback(
            local_rank=local_rank,
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
