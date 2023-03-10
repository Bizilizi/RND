import datetime
import logging
import typing as t

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers

import wandb
from avalanche.logging import InteractiveLogger
from src.avalanche.configuration.config import BaseTrainConfig
from src.avalanche.loggers.interactive_wandb import InteractiveWandBLogger

# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def add_arguments(parser):
    """
    This function extends parser with extra training parameters
    They are general and serve all models in mono repo

    Model specific parameter will be injected later from config file
    """

    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--resume_from",
        nargs="?",
        type=str,
        help="path to resume model",
        default=None,
    )
    parser.add_argument(
        "--experiment_name",
        nargs="?",
        type=str,
        help="Name of experiment",
        default=None,
    )
    parser.add_argument(
        "--seed",
        nargs="?",
        type=int,
        help="seed for random",
        default=42,
    )
    parser.add_argument(
        "--run_id",
        nargs="?",
        type=str,
        help="wandb run id",
        default=None,
    )
    parser.add_argument(
        "--wandb_dir",
        nargs="?",
        type=str,
        help="wandb run id",
        default=".wandb",
    )
    parser.add_argument(
        "--group",
        nargs="?",
        type=str,
        help="wandb run group",
        default=None,
    )
    parser.add_argument(
        "--sweep_id",
        nargs="?",
        type=str,
        help="wandb sweep id",
        default=None,
    )

    return parser


def get_loggers(
    config: BaseTrainConfig,
    model: pl.LightningModule,
    wandb_params: t.Optional[t.Dict[str, t.Any]] = None,
):
    """
    Returns validation (avalanche) and training (pytorch-lightning) loggers
    """

    # Create Evaluation plugin
    evaluation_loggers = []
    if config.evaluation_logger == "wandb":
        evaluation_loggers.append(
            InteractiveWandBLogger(
                project_name=wandb_params["project"],
                run_name=wandb_params["name"],
                config=wandb_params["config"],
                params=wandb_params,
            )
        )
    elif config.evaluation_logger in ["interactive", "int"]:
        evaluation_loggers.append(InteractiveLogger())

    # Create avalanche strategy
    if config.train_logger == "wandb":

        train_logger = pl_loggers.WandbLogger(
            project=wandb_params["project"],
            log_model=False,
            experiment=wandb.run,
        )
        train_logger.watch(model)

    elif config.train_logger == "tensorboard":
        train_logger = pl_loggers.TensorBoardLogger(save_dir=config.logging_path)
    else:
        train_logger = None

    return train_logger, evaluation_loggers


def get_device(config):
    if config.accelerator == "gpu":
        return torch.device("cuda")
    elif config.accelerator == "mps":
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_wandb_params(args, config):
    """
    Construct wandb parameters based on passed arguments and config
    """

    wandb_params = dict(
        project=args.model.upper(),
        id=args.run_id,
        entity="vgg-continual-learning",
        group=args.group,
        dir=args.wandb_dir,
    )
    wandb.init(**wandb_params)

    # Override config with sweep
    for k, v in wandb.config.items():
        if k == "accelerator":
            continue
        setattr(config, k, v)

    wandb.config.update(dict(config))
    wandb_params["config"] = wandb.config

    return wandb_params
