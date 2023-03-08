import datetime
import logging
import typing as t

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers

import wandb
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from src.avalanche.configuration.config import BaseTrainConfig
from src.avalanche.loggers.interactive_wandb import InteractiveWandBLogger
from src.rnd.configuration.config import TrainConfig as RNDConfig
from src.rnd.init_scripts import get_callbacks as rnd_get_callbacks
from src.rnd.init_scripts import get_evaluation_plugin as rnd_get_evaluation_plugin
from src.rnd.init_scripts import get_model as rnd_get_model_and_device
from src.vae_ft.configuration.config import TrainConfig as VAEFtConfig
from src.vae_ft.init_scrips import get_callbacks as vae_ft_get_callbacks
from src.vae_ft.init_scrips import get_evaluation_plugin as vae_ft_get_evaluation_plugin
from src.vae_ft.init_scrips import get_model as vae_ft_get_model
from src.vq_vae.init_scrips import get_callbacks as vq_vae_get_callbacks
from src.vq_vae.init_scrips import get_evaluation_plugin as vq_vae_get_evaluation_plugin
from src.vq_vae.init_scrips import get_model as vq_vae_get_model
from src.vq_vae.configuration.config import TrainConfig as VqVaeConfig

# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def add_arguments(parser):
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


def get_typed_config(args, ini_config) -> BaseTrainConfig:
    assert args.model, "You have to specify what model to train by '--model' parameter"
    if args.model == "rnd":
        config = RNDConfig.construct_typed_config(ini_config)
    elif args.model == "vae-ft":
        config = VAEFtConfig.construct_typed_config(ini_config)
    elif args.model == "vq-vae":
        config = VqVaeConfig.construct_typed_config(ini_config)
    else:
        assert False, "Unknown value '--model' parameter"

    return config


def get_loggers(
    config: BaseTrainConfig,
    model: pl.LightningModule,
    wandb_params: t.Optional[t.Dict[str, t.Any]] = None,
):
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


def get_model_and_device(args, config: t.Union[RNDConfig, VAEFtConfig]):
    # Get device
    if config.accelerator == "gpu":
        device = torch.device("cuda")
    elif config.accelerator == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Create CL strategy
    assert args.model, "You have to specify what model to train by '--model' parameter"
    if args.model == "rnd":
        model = rnd_get_model_and_device(config, device)
    elif args.model == "vae-ft":
        model = vae_ft_get_model(config, device)
    elif args.model == "vq-vae":
        model = vq_vae_get_model(config, device)
    else:
        assert False, "Unknown value '--model' parameter"

    return model, device


def get_evaluation_plugin(
    args, benchmark, evaluation_loggers, is_using_wandb
) -> t.Optional[EvaluationPlugin]:
    # Create CL strategy
    assert args.model, "You have to specify what model to train by '--model' parameter"
    if args.model == "rnd":
        eval_plugin = rnd_get_evaluation_plugin(
            benchmark,
            evaluation_loggers=evaluation_loggers,
            is_using_wandb=is_using_wandb,
        )
    elif args.model == "vae-ft":
        eval_plugin = vae_ft_get_evaluation_plugin(evaluation_loggers)
    elif args.model == "vq-vae":
        eval_plugin = vq_vae_get_evaluation_plugin(evaluation_loggers)
    else:
        assert False, "Unknown value '--model' parameter"

    return eval_plugin


def get_callbacks(args, config):
    # Create CL strategy
    assert args.model, "You have to specify what model to train by '--model' parameter"
    if args.model == "rnd":
        return rnd_get_callbacks()
    elif args.model == "vae-ft":
        return vae_ft_get_callbacks(config=config)
    elif args.model == "vq-vae":
        return vq_vae_get_callbacks(config=config)
    else:
        assert False, "Unknown value '--model' parameter"


def get_experiment_name(args, config):
    assert args.model, "You have to specify what model to train by '--model' parameter"
    if args.model == "vae-ft":
        return (
            f"{config.model_backbone.upper()}."
            f"RI-{config.num_random_images}."
            f"RN-{config.num_random_noise}."
            f"Dr-{config.regularization_dropout}."
            f"Wd-{config.regularization_lambda}."
        )

    return f"CL-train. Time: {datetime.datetime.now():%Y-%m-%d %H:%M}"
