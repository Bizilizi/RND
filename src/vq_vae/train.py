import pathlib
import shutil
from configparser import ConfigParser

import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.classification_dataset import (
    ClassificationDataset,
    make_classification_dataset,
)
from torch.utils.data import ConcatDataset
from torchvision import transforms

import wandb
from avalanche.benchmarks import SplitCIFAR10, SplitImageNet

from src.avalanche.strategies import NaivePytorchLightning
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config_with_args
from src.vq_vae.callbacks.reconstruction_visualization_plugin import (
    ReconstructionVisualizationPlugin,
)
from src.vq_vae.configuration.config import TrainConfig
from src.vq_vae.init_scrips import get_callbacks, get_evaluation_plugin, get_model
from src.vq_vae.train_classifier import (
    train_classifier_on_all_classes,
    train_classifier_on_observed_only_classes,
)
from src.vq_vae.train_image_gpt import (
    train_img_gpt_on_observed_only_classes,
    bootstrap_dataset,
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

    image_gpt = None

    for train_experience, test_experience in zip(
        benchmark.train_stream, benchmark.test_stream
    ):
        train_dataset = train_experience.dataset
        val_dataset = test_experience.dataset

        # bootstrap old data
        if (
            cl_strategy.experience_step != 0
            and config.num_random_images != 0
            and image_gpt is not None
        ):
            bootstrapped_dataset = bootstrap_dataset(
                image_gpt=image_gpt,
                vq_vae_model=cl_strategy.model,
                num_images=config.num_random_images * cl_strategy.experience_step,
            )
            train_experience.dataset = train_dataset + bootstrapped_dataset

        # Train VQ-VAE
        with torch.autograd.set_detect_anomaly(True):
            cl_strategy.train(
                train_experience,
                [test_experience],
            )

        # Train new image gpt model
        image_gpt = train_img_gpt_on_observed_only_classes(
            strategy=cl_strategy,
            config=config,
            train_dataset=train_dataset,
            test_dataset=val_dataset,
            device=device,
        )

        # Evaluate VQ-VAE and linear classifier
        # cl_strategy.eval(benchmark.test_stream)
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
            f"VQL-{config.vq_loss_weight:0.3f} | "
            f"ReL-{config.reconstruction_loss_weight:0.3f} | "
            f"DL-{config.downstream_loss_weight:0.3f}"
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
        n_experiences=5,
        return_task_id=True,
        shuffle=True,
        dataset_root=config.dataset_path,
        train_transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (1.0, 1.0, 1.0)),
            ]
        ),
        eval_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (1.0, 1.0, 1.0)),
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
