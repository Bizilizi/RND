import argparse
import typing as t
from configparser import ConfigParser

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything

from src.configuration.config import TrainConfig


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
) -> None:
    # Create DataModule
    datamodule = ...
    datamodule.setup()

    # Instantiate model
    model = ...

    # Create logger
    if config.logger_type == "wandb":
        wandb.init(
            project="RND", id=run_id, entity="ewriji", config=dict(config)
        )
        logger = pl_loggers.WandbLogger(project="RND", log_model="all")
        logger.watch(model)
    elif config.logger_type == "tensorboard":
        logger = pl_loggers.TensorBoardLogger(save_dir=config.logging_path)
    else:
        logger = None

    # Training
    trainer = Trainer(
        gpus=config.gpus,
        check_val_every_n_epoch=config.validate_every_n,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=config.max_epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_from)


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
        train_loop(config, args.resume_from, args.run_id)
    except KeyboardInterrupt:
        print("Training successfully interrupted.")
