import argparse
import datetime
import logging
import os
import typing as t
from configparser import ConfigParser

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from torchvision.transforms import ToTensor

import wandb
from avalanche.benchmarks import SplitMNIST
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from src.avalanche.configuration.config import BaseTrainConfig
from src.avalanche.loggers.interactive_wandb import InteractiveWandBLogger
from src.avalanche.strategies import NaivePytorchLightning
from src.rnd.configuration.config import TrainConfig as RNDConfig
from src.rnd.init_scripts import get_callbacks as rnd_get_callbacks
from src.rnd.init_scripts import get_evaluation_plugin as rnd_get_evaluation_plugin
from src.rnd.init_scripts import get_model as rnd_get_model_and_device
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config_with_args, parse_arguments
from src.vae_ft.configuration.config import TrainConfig as VAEFtConfig
from src.vae_ft.init_scrips import get_callbacks as vae_ft_get_callbacks
from src.vae_ft.init_scrips import get_evaluation_plugin as vae_ft_get_evaluation_plugin
from src.vae_ft.init_scrips import get_model as vae_ft_get_model

# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def train_loop(
    benchmark: t.Union[SplitMNIST],
    cl_strategy: NaivePytorchLightning,
    is_using_wandb: bool,
) -> None:
    """
    :return:
    """

    results = []
    for train_experience, test_experience in zip(
        benchmark.train_stream, benchmark.test_stream
    ):
        cl_strategy.train(train_experience, [test_experience])
        results.append(cl_strategy.eval(benchmark.test_stream))

    if is_using_wandb:
        log_summary_table_to_wandb(benchmark.train_stream, benchmark.test_stream)


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

    return parser


def get_typed_config(args, ini_config) -> BaseTrainConfig:
    assert args.model, "You have to specify what model to train by '--model' parameter"
    if args.model == "rnd":
        config = RNDConfig.construct_typed_config(ini_config)
    elif args.model == "vae-ft":
        config = VAEFtConfig.construct_typed_config(ini_config)
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
    else:
        assert False, "Unknown value '--model' parameter"


def main(args):
    os.environ["WANDB_START_METHOD"] = "thread"

    # Make it deterministic
    seed_everything(args.seed)

    # Reading configuration from ini file
    assert (
        args.config
    ), "Please fill the --config argument with valid path to configuration file."
    ini_config = ConfigParser()
    ini_config.read(args.config)

    # Unpack config to typed config class and create the model
    config = get_typed_config(args=args, ini_config=ini_config)
    overwrite_config_with_args(args, config)

    # Generate experiment name if necessary
    if args.experiment_name is None:
        args.experiment_name = (
            f"CL-train. Time: {datetime.datetime.now():%Y-%m-%d %H:%M}"
        )

    # Construct wandb params if necessary
    is_using_wandb = (
        config.train_logger == "wandb" or config.evaluation_logger == "wandb"
    )
    if is_using_wandb:
        wandb_params = dict(
            project=args.model.upper(),
            id=args.run_id,
            entity="vgg-continual-learning",
            config=dict(config),
            name=args.experiment_name,
        )
        wandb.init(**wandb_params)
    else:
        wandb_params = None

    # Create benchmark, model and loggers
    benchmark = SplitMNIST(
        n_experiences=5,
        seed=args.seed,
        train_transform=ToTensor(),
        eval_transform=ToTensor(),
        shuffle=False,
    )
    model, device = get_model_and_device(args, config)

    # Create evaluation plugin and train/val loggers
    cl_strategy_logger, eval_plugin_loggers = get_loggers(config, model, wandb_params)
    evaluation_plugin = get_evaluation_plugin(
        args,
        benchmark=benchmark,
        evaluation_loggers=eval_plugin_loggers,
        is_using_wandb=is_using_wandb,
    )

    cl_strategy = NaivePytorchLightning(
        accelerator=config.accelerator,
        devices=config.devices,
        validate_every_n=config.validate_every_n,
        accumulate_grad_batches=config.accumulate_grad_batches,
        train_logger=cl_strategy_logger,
        initial_resume_from=args.resume_from,
        model=model,
        device=device,
        optimizer=model.configure_optimizers(),
        criterion=model.criterion,
        train_mb_size=config.batch_size,
        train_mb_num_workers=config.num_workers,
        train_epochs=config.max_epochs,
        eval_mb_size=config.batch_size,
        evaluator=evaluation_plugin,
        callbacks=get_callbacks(args, config),
        max_epochs=config.max_epochs,
        min_epochs=config.min_epochs,
        restore_best_model=True,
    )

    # Run training process
    print(f"Running training process..")
    try:
        train_loop(
            benchmark=benchmark, cl_strategy=cl_strategy, is_using_wandb=is_using_wandb
        )
    except KeyboardInterrupt:
        print("Training successfully interrupted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model trainer")
    parser = add_arguments(parser)
    args = parse_arguments(parser)

    main(args)
