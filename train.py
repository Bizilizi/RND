import argparse
import datetime
import typing as t
from configparser import ConfigParser

import avalanche.logging as av_loggers
import wandb
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    confusion_matrix_metrics,
    cpu_usage_metrics,
    disk_usage_metrics,
    forgetting_metrics,
    loss_metrics,
    timing_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from torchvision.datasets import MNIST

from src.configuration.config import TrainConfig
from src.model.simple_mlp import PLSimpleMLP
from src.scenarios.mnist import NormalPermutedMNIST
from src.strategies.naive_pl import NaivePytorchLightning


def overwrite_config(cli_args, config):
    """The method overwrites config fields with passed to cli arguments"""

    for k, v in vars(cli_args).items():
        if hasattr(config, k):
            setattr(config, k, type(getattr(config, k))(v))


def parse_arguments(parser):
    """
    This method constructs a new parser for cli command with new unregisters
    arguments with str type and runs `parse_args` on it.
    """

    parsed, unknown = parser.parse_known_args()

    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split("=")[0], type=str)

    args = parser.parse_args()
    return args


def train_loop(
    config: TrainConfig,
    experiment_name: str,
    resume_from: t.Optional[str] = None,
    run_id: t.Optional[str] = None,
) -> None:
    """

    :param config: Train config
    :param resume_from: Path to checkpoint with model weights
    :param run_id: Weight&Biases id of the run
    :param experiment_name: Name of experiment
    :return:
    """

    # Construct wandb params if necessary
    if config.train_logger == "wandb" or config.train_logger == "wandb":
        wandb_params = dict(
            project="RND",
            id=run_id,
            entity="ewriji",
            config=dict(config),
            name=experiment_name,
        )
    else:
        wandb_params = None

    # Create benchmark
    benchmark = NormalPermutedMNIST()
    model = PLSimpleMLP(learning_rate=0.005, num_classes=10)

    # Create Evaluation plugin
    evaluation_loggers = []
    if config.train_logger == "wandb":
        evaluation_loggers.append(
            av_loggers.WandBLogger(
                project_name=wandb_params["project"],
                run_name=wandb_params["name"],
                config=wandb_params["config"],
                params=wandb_params,
            )
        )
    elif config.train_logger == "interactive":
        evaluation_loggers.append(InteractiveLogger())

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(
            num_classes=benchmark.n_classes, save_image=False, stream=True
        ),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=evaluation_loggers,
    )

    # Create avalanche strategy
    if config.train_logger == "wandb":
        if wandb.run is None:
            wandb.init(**wandb_params)

        train_logger = pl_loggers.WandbLogger(
            project=wandb_params["project"],
            log_model="all",
        )
        train_logger.watch(model)
    elif config.train_logger == "tensorboard":
        train_logger = pl_loggers.TensorBoardLogger(save_dir=config.logging_path)
    else:
        train_logger = None

    cl_strategy = NaivePytorchLightning(
        config=config,
        train_logger=train_logger,
        resume_from=resume_from,
        model=model,
        optimizer=model.configure_optimizers(),
        criterion=model.loss,
        train_mb_size=100,
        train_epochs=4,
        eval_mb_size=100,
        evaluator=eval_plugin,
    )

    results = []
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        results.append(cl_strategy.eval(benchmark.test_stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model trainer")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
        default="./src/configuration/train.ini",
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
    args = parse_arguments(parser)

    # Make it deterministic
    seed_everything(args.seed)

    # Reading configuration from ini file
    ini_config = ConfigParser()
    ini_config.read(args.config)

    # Unpack config to typed config class
    config = TrainConfig.construct_typed_config(ini_config)
    overwrite_config(args, config)

    # Generate experiment name if necessary
    if args.experiment_name is None:
        args.experiment_name = f"CL-train-{datetime.datetime.now()}"

    # Run training process
    print(f"Running training process..")
    try:
        train_loop(
            config=config,
            experiment_name=args.experiment_name,
            resume_from=args.resume_from,
            run_id=args.run_id,
        )
    except KeyboardInterrupt:
        print("Training successfully interrupted.")
