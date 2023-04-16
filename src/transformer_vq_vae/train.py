import pathlib
import shutil
from configparser import ConfigParser

import torch
from torchvision import transforms

import wandb
from avalanche.benchmarks import SplitCIFAR10, SplitImageNet

from src.avalanche.strategies import NaivePytorchLightning
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config_with_args
from src.transformer_vq_vae.callbacks.reconstruction_visualization_plugin import (
    ReconstructionVisualizationPlugin,
)
from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.init_scrips import (
    get_callbacks,
    get_evaluation_plugin,
    get_model,
)
from src.transformer_vq_vae.train_classifier import (
    train_classifier_on_all_classes,
    train_classifier_on_observed_only_classes,
)
from train_utils import get_device, get_loggers, get_wandb_params


def train_loop(
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
        # Train VQ-VAE
        cl_strategy.train(train_experience, [test_experience])

        # Train linear classifier, but before we freeze model params
        # We train two classifiers. One to predict all classes,
        # another to predict only observed so far classes.
        cl_strategy.model.freeze()

        train_classifier_on_all_classes(
            strategy=cl_strategy, config=config, benchmark=benchmark, device=device
        ).to(device)
        observed_only_clf_head = train_classifier_on_observed_only_classes(
            strategy=cl_strategy, config=config, benchmark=benchmark, device=device
        ).to(device)

        cl_strategy.model.set_clf_head(observed_only_clf_head)

        # Evaluate VQ-VAE and linear classifier
        cl_strategy.eval(benchmark.test_stream)

        # Reset linear classifier and unfreeze params
        cl_strategy.model.reset_clf_head()
        cl_strategy.model.unfreeze()

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
            f"CL-{config.contrastive_loss_loss_weight:0.3f} | "
            f"PxLM-{config.decoder_regression_loss_loss_weight:0.3f} | "
            f"ZLML-{config.encoder_mlm_loss_loss_weight:0.3f}"
        )
        wandb_params["name"] = wandb.run.name
    else:
        wandb_params = None

    # Create benchmark, model and loggers
    # datasets_dir = pathlib.Path(config.dataset_path)
    # target_dataset_dir = pathlib.Path("/tmp/dzverev_data/")
    # target_dataset_dir.mkdir(exist_ok=True)
    #
    # zip_path = datasets_dir / "cifar-10-python.tar.gz"
    # dataset_path = datasets_dir / "cifar-10-batches-py"
    #
    # target_zip_path = target_dataset_dir / "cifar-10-python.tar.gz"
    # target_dataset_path = target_dataset_dir / "cifar-10-batches-py"
    #
    # if zip_path.exists() and not target_zip_path.exists():
    #     shutil.copy(str(zip_path), str(target_zip_path))
    #
    # if dataset_path.exists() and not target_dataset_path.exists():
    #     shutil.copytree(str(dataset_path), str(target_dataset_path))

    benchmark = SplitCIFAR10(
        n_experiences=1,
        return_task_id=True,
        shuffle=True,
        dataset_root=config.dataset_path,
        train_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
            ]
        ),
        eval_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
            ]
        ),
    )

    device = get_device(config)
    model = get_model(config, device)

    # Create evaluation plugin and train/val loggers
    cl_strategy_logger, eval_plugin_loggers = get_loggers(config, model, wandb_params)
    evaluation_plugin = get_evaluation_plugin(
        benchmark, eval_plugin_loggers, is_using_wandb
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
        callbacks=get_callbacks(config),
        max_epochs=config.max_epochs,
        min_epochs=config.min_epochs,
        best_model_path_prefix=config.best_model_prefix,
        plugins=[ReconstructionVisualizationPlugin(num_tasks_in_batch=2)],
    )

    # Run training process
    print(f"Running training process..")
    try:
        train_loop(
            benchmark=benchmark,
            cl_strategy=cl_strategy,
            is_using_wandb=is_using_wandb,
            config=config,
            device=device,
        )
    except KeyboardInterrupt:
        print("Training successfully interrupted.")
