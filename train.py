import argparse
import datetime
import logging
import typing as t
from collections import defaultdict
from configparser import ConfigParser

import torch
from avalanche.benchmarks import SplitMNIST
from avalanche.evaluation.metrics import timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything

import wandb
from src.configuration.config import TrainConfig
from src.loggers.interactive_wandb import InteractiveWandBLogger
from src.metrics.rnd_accuracy import rnd_accuracy_metrics
from src.metrics.rnd_confusion_matrix import rnd_confusion_matrix_metrics
from src.metrics.rnd_forgetting import rnd_forgetting_metrics
from src.metrics.rnd_loss import rnd_loss_metrics
from src.model.rnd.gan_generator import MNISTGanGenerator
from src.model.rnd.rnd import RND
from src.strategies.naive_pl import NaivePytorchLightning
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config, parse_arguments

# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def train_loop(
    config: TrainConfig,
    experiment_name: str,
    resume_from: t.Optional[str] = None,
    run_id: t.Optional[str] = None,
    seed: int = 42,
) -> None:
    """

    :param config: Train config
    :param resume_from: Path to checkpoint with model weights
    :param run_id: Weight&Biases id of the run
    :param experiment_name: Name of experiment
    :return:
    """
    # Get device
    if config.accelerator == "gpu":
        device = torch.device("cuda")
    elif config.accelerator == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Construct wandb params if necessary
    is_using_wandb = (
        config.train_logger == "wandb" or config.evaluation_logger == "wandb"
    )
    if is_using_wandb:
        wandb_params = dict(
            project="RND",
            id=run_id,
            entity="vgg-continual-learning",
            config=dict(config),
            name=experiment_name,
        )
        wandb.init(**wandb_params)
    else:
        wandb_params = None

    # Create benchmark and model
    benchmark = SplitMNIST(n_experiences=10, seed=seed)

    assert config.generator_checkpoint, "Generator checkpoint is necessary to provide!"
    generator = MNISTGanGenerator(input_dim=config.input_dim, output_dim=784)
    generator.load_state_dict(
        torch.load(config.generator_checkpoint, map_location=torch.device("cpu"))
    )
    generator.to(device)

    model = RND(
        generator=generator,
        num_random_images=config.num_random_images,
        l2_threshold=config.l2_threshold,
        rnd_latent_dim=config.rnd_latent_dim,
        num_classes=10,
        num_generation_attempts=20,
    )

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
    elif config.evaluation_logger == "interactive":
        evaluation_loggers.append(InteractiveLogger())

    eval_plugin = EvaluationPlugin(
        timing_metrics(epoch_running=True),
        rnd_forgetting_metrics(experience=True, stream=True, accuracy=True),
        rnd_accuracy_metrics(experience=True, stream=True),
        rnd_confusion_matrix_metrics(
            num_classes=benchmark.n_classes,
            save_image=False,
            stream=True,
            wandb=is_using_wandb,
        ),
        rnd_loss_metrics(experience=True, stream=True),
        suppress_warnings=True,
        loggers=evaluation_loggers,
    )

    # Create avalanche strategy
    if config.train_logger == "wandb":

        train_logger = pl_loggers.WandbLogger(
            project=wandb_params["project"],
            log_model=False,
            experiment=wandb.run,
        )
        train_logger.watch(model)

        # Add CL step metric to wandb
        # train_logger.experiment.define_metric("trainer/experience_step")
    elif config.train_logger == "tensorboard":
        train_logger = pl_loggers.TensorBoardLogger(save_dir=config.logging_path)
    else:
        train_logger = None

    cl_strategy = NaivePytorchLightning(
        accelerator=config.accelerator,
        devices=config.devices,
        validate_every_n=config.validate_every_n,
        accumulate_grad_batches=config.accumulate_grad_batches,
        train_logger=train_logger,
        resume_from=resume_from,
        model=model,
        device=device,
        optimizer=model.configure_optimizers(),
        criterion=model.criterion,
        train_mb_size=config.batch_size,
        train_mb_num_workers=config.num_workers,
        train_epochs=config.max_epochs,
        eval_mb_size=config.batch_size,
        evaluator=eval_plugin,
    )

    results = []
    for train_experience, test_experience in zip(
        benchmark.train_stream, benchmark.test_stream
    ):
        if train_experience.current_experience == 0:
            model.keep_sampling = False
        else:
            model.keep_sampling = True

        cl_strategy.train(train_experience, [test_experience])
        results.append(cl_strategy.eval(benchmark.test_stream))

    if is_using_wandb:
        log_summary_table_to_wandb(benchmark.train_stream, benchmark.test_stream)


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
        args.experiment_name = (
            f"CL-train. Time: {datetime.datetime.now():%Y-%m-%d %H:%M}"
        )

    # Run training process
    print(f"Running training process..")
    try:
        train_loop(
            config=config,
            experiment_name=args.experiment_name,
            resume_from=args.resume_from,
            run_id=args.run_id,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        print("Training successfully interrupted.")
