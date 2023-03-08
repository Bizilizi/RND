import argparse
import logging
import typing as t
from configparser import ConfigParser
from functools import partial

from pytorch_lightning import seed_everything
from torchvision.transforms import ToTensor

import wandb
from avalanche.benchmarks import SplitMNIST, SplitCIFAR10
from src.avalanche.strategies import NaivePytorchLightning
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config_with_args, parse_arguments
from train_utils import (
    add_arguments,
    get_callbacks,
    get_evaluation_plugin,
    get_experiment_name,
    get_loggers,
    get_model_and_device,
    get_typed_config,
)

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

    for train_experience, test_experience in zip(
        benchmark.train_stream, benchmark.test_stream
    ):
        cl_strategy.train(train_experience, [test_experience])
        cl_strategy.eval(benchmark.test_stream)

    if is_using_wandb:
        log_summary_table_to_wandb(benchmark.train_stream, benchmark.test_stream)


def main(args):
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

    # Construct wandb params if necessary
    is_using_wandb = (
        config.train_logger == "wandb"
        or config.evaluation_logger == "wandb"
        or args.run_id
    )
    if is_using_wandb:
        wandb_params = dict(
            project=args.model.upper(),
            id=args.run_id,
            entity="vgg-continual-learning",
            group=args.group,
            dir=args.wandb_dir,
        )
        wandb.init(**wandb_params)

        # Override config with sweep
        if wandb.config:
            for k, v in wandb.config.items():
                if k == "accelerator":
                    continue
                setattr(config, k, v)

            wandb.config.update(dict(config))
            wandb.run.name = args.experiment_name or get_experiment_name(args, config)

        wandb_params["name"] = wandb.run.name
        wandb_params["config"] = wandb.config
    else:
        wandb_params = None

    # Create benchmark, model and loggers
    benchmark = SplitCIFAR10(
        n_experiences=5,
        seed=args.seed,
        train_transform=ToTensor(),
        eval_transform=ToTensor(),
        shuffle=False,
        dataset_root=config.dataset_path,
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
        best_model_path_prefix=config.best_model_prefix,
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

    # If we run training in sweep mode
    if args.sweep_id:
        wandb.agent(
            args.sweep_id,
            partial(main, args),
            project=args.model.upper(),
        )
    else:
        main(args)
