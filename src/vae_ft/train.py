import datetime
import typing as t
from configparser import ConfigParser

from pytorch_lightning import seed_everything
from torchvision.transforms import ToTensor

import wandb
from avalanche.benchmarks import SplitMNIST
from src.avalanche.strategies import NaivePytorchLightning
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config_with_args
from src.vae_ft.configuration.config import TrainConfig
from src.vae_ft.init_scrips import get_callbacks, get_evaluation_plugin, get_model
from train_utils import get_device, get_loggers, get_wandb_params


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
        cl_strategy.experience_step += 1

    if is_using_wandb:
        log_summary_table_to_wandb(benchmark.train_stream, benchmark.test_stream)


def main(args):

    # Reading configuration from ini file
    assert (
        args.config
    ), "Please fill the --config argument with valid path to configuration file."
    ini_config = ConfigParser()
    ini_config.read(args.config)

    # Unpack config to typed config class and create the model
    config = TrainConfig.construct_typed_config(ini_config)

    overwrite_config_with_args(args, config)

    # Construct wandb params if necessary
    is_using_wandb = (
        config.train_logger == "wandb"
        or config.evaluation_logger == "wandb"
        or args.run_id
    )
    if is_using_wandb:
        wandb_params = get_wandb_params(args, config)

        wandb.run.name = args.experiment_name or (
            f"{config.model_backbone.upper()}."
            f"RI-{config.num_random_images}."
            f"RN-{config.num_random_noise}."
            f"Dr-{config.regularization_dropout}."
            f"Wd-{config.regularization_lambda}."
        )
        wandb_params["name"] = wandb.run.name
    else:
        wandb_params = None

    # Fix path params
    today = datetime.datetime.now()
    run_id = wandb_params["id"] if wandb_params else today.strftime("%Y_%m_%d_%H_%M")

    config.checkpoint_path += f"/{run_id}/model"
    config.best_model_prefix += f"/{run_id}/best_model"

    # Create benchmark, model and loggers
    benchmark = SplitMNIST(
        n_experiences=1,
        seed=args.seed,
        train_transform=ToTensor(),
        eval_transform=ToTensor(),
        shuffle=False,
        dataset_root=config.dataset_path,
    )
    device = get_device(config)
    model = get_model(config, device)

    # Create evaluation plugin and train/val loggers
    cl_strategy_logger, eval_plugin_loggers = get_loggers(config, model, wandb_params)
    evaluation_plugin = get_evaluation_plugin(eval_plugin_loggers)

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
        callbacks=get_callbacks(config),
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
