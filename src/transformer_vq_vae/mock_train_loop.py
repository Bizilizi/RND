import datetime
import os
import pathlib
import shutil
from configparser import ConfigParser

import torch
from torchvision import transforms

import wandb
from avalanche.benchmarks import SplitCIFAR10
from src.avalanche.strategies import NaivePytorchLightning
from src.transformer_vq_vae.callbacks.reconstruction_visualization_plugin import (
    ReconstructionVisualizationPlugin,
)
from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.init_scrips import (
    get_callbacks,
    get_evaluation_plugin,
    get_model,
    get_train_plugins,
)
from src.transformer_vq_vae.model_future import model_future_samples
from src.transformer_vq_vae.train_classifier import (
    train_classifier_on_all_classes,
    train_classifier_on_observed_only_classes,
)
from src.transformer_vq_vae.train_image_gpt import bootstrap_past_samples, train_igpt
from src.transformer_vq_vae.utils.wrap_empty_indices import (
    wrap_dataset_with_empty_indices,
)
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config_with_args
from train_utils import get_device, get_loggers, get_wandb_params

from pathlib import Path


def mock_train_loop(
    benchmark: SplitCIFAR10,
    cl_strategy: NaivePytorchLightning,
    is_using_wandb: bool,
    config: TrainConfig,
    device: torch.device,
) -> None:
    """
    :return:
    """

    for train_experience, test_experience in zip(
        benchmark.train_stream, benchmark.test_stream
    ):
        train_experience.dataset = wrap_dataset_with_empty_indices(
            train_experience.dataset, num_neighbours=config.quantize_top_k
        )
        test_experience.dataset = wrap_dataset_with_empty_indices(
            test_experience.dataset, num_neighbours=config.quantize_top_k
        )
        igpt_train_dataset = train_experience.dataset + test_experience.dataset

        # Train VQ-VAE
        cl_strategy.train(train_experience, [test_experience])

        cl_strategy.experience_step += 1

    if is_using_wandb:
        log_summary_table_to_wandb(benchmark.train_stream, benchmark.test_stream)
