import argparse
import typing as t
from configparser import ConfigParser

import wandb
from avalanche.benchmarks import SplitMNIST
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

from src.configuration.config import TrainConfig
from src.model.simple_mlp import PLSimpleMLP
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
    resume_from: t.Optional[str] = None,
    run_id: t.Optional[str] = None,
    seed: int = 42,
) -> None:
    # Create benchmark
    benchmark = SplitMNIST(n_experiences=5, seed=seed)

    # Instantiate model
    model = PLSimpleMLP(learning_rate=0.005, num_classes=10)

    # Create loggers
    if config.logger_type == "wandb":
        wandb.init(project="RND", id=run_id, entity="ewriji", config=dict(config))
        logger = pl_loggers.WandbLogger(project="RND", log_model="all")
        logger.watch(model)
    elif config.logger_type == "tensorboard":
        logger = pl_loggers.TensorBoardLogger(save_dir=config.logging_path)
    else:
        logger = None

    interactive_logger = InteractiveLogger()

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
        loggers=[interactive_logger],
    )

    # Create avalanche strategy
    cl_strategy = NaivePytorchLightning(
        config=config,
        train_logger=logger,
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

    # Run training process
    print(f"Running training process..")
    try:
        train_loop(config, args.resume_from, args.run_id, args.seed)
    except KeyboardInterrupt:
        print("Training successfully interrupted.")
