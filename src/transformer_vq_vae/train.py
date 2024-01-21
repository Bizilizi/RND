import datetime
import os
import pathlib
import shutil
from configparser import ConfigParser

import torch
from torchvision import transforms

import wandb
from avalanche.benchmarks import SplitCIFAR10, SplitCIFAR100, NCExperience
from src.avalanche.strategies import NaivePytorchLightning
from src.transformer_vq_vae.callbacks.reconstruction_visualization_plugin import (
    ReconstructionVisualizationPlugin,
)
from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.data.tiny_imagenet import SplitTinyImageNet
from src.transformer_vq_vae.init_scrips import (
    get_callbacks,
    get_evaluation_plugin,
    get_model,
    get_train_plugins,
)
from src.transformer_vq_vae.mock_train_loop import mock_train_loop
from src.transformer_vq_vae.model_future import model_future_samples
from src.transformer_vq_vae.train_classifier import (
    train_classifier_on_all_classes,
    train_classifier_on_observed_only_classes,
)
from src.transformer_vq_vae.train_image_gpt import bootstrap_past_samples, train_igpt
from src.transformer_vq_vae.utils.wrap_empty_indices import (
    convert_avalanche_dataset_to_vq_mae_dataset,
)
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config_with_args
from train_utils import get_device, get_loggers, get_wandb_params

from pathlib import Path


def get_benchmark(config: TrainConfig, target_dataset_dir):
    if config.dataset == "cifar10":
        config.image_size = 32
        return SplitCIFAR10(
            n_experiences=config.num_tasks,
            return_task_id=True,
            shuffle=True,
            dataset_root=target_dataset_dir,
            train_transform=transforms.Compose(
                [
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
    elif config.dataset == "cifar100":
        config.image_size = 32
        return SplitCIFAR100(
            n_experiences=config.num_tasks,
            return_task_id=True,
            shuffle=True,
            dataset_root=target_dataset_dir,
            train_transform=transforms.Compose(
                [
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
    elif config.dataset == "tiny-imagenet":
        config.image_size = 64
        return SplitTinyImageNet(
            n_experiences=config.num_tasks,
            return_task_id=True,
            shuffle=True,
            dataset_root=target_dataset_dir,
            train_transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
            eval_transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        )


def get_num_random_past_samples(
    config: TrainConfig, cl_strategy: NaivePytorchLightning
):
    if config.num_random_past_samples_schedule == "fixed":
        return config.num_random_past_samples

    if config.num_random_past_samples_schedule == "linear":
        return config.num_random_past_samples * cl_strategy.experience_step

    if config.num_random_past_samples_schedule == "schedule":
        schedule = torch.linspace(0, config.num_random_past_samples, config.num_tasks)
        return int(schedule[int(cl_strategy.experience_step)])


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

        train_experience.dataset = convert_avalanche_dataset_to_vq_mae_dataset(
            train_experience.dataset,
            num_neighbours=config.quantize_top_k,
            config=config,
            time_tag=0,
        )
        test_experience.dataset = convert_avalanche_dataset_to_vq_mae_dataset(
            test_experience.dataset,
            num_neighbours=config.quantize_top_k,
            config=config,
            time_tag=0,
        )
        igpt_train_dataset = train_experience.dataset + test_experience.dataset

        # Bootstrap old data and modeled future samples
        if cl_strategy.experience_step != 0 and image_gpt is not None:
            image_gpt.to(device)
            cl_strategy.model.to(device)

            if config.num_random_past_samples != 0:
                print(f"Bootstrap vae model..")
                bootstrapped_dataset = bootstrap_past_samples(
                    image_gpt=image_gpt,
                    vq_vae_model=cl_strategy.model,
                    num_images=get_num_random_past_samples(config, cl_strategy),
                    dataset_path=config.bootstrapped_dataset_path,
                    config=config,
                    experience_step=cl_strategy.experience_step,
                )

                train_experience.dataset = (
                    train_experience.dataset + bootstrapped_dataset
                )

                igpt_train_dataset = igpt_train_dataset + bootstrapped_dataset

            if config.num_random_future_samples != 0:
                print(f"Model future samples..")
                future_dataset = model_future_samples(
                    vq_vae_model=cl_strategy.model,
                    num_images=(
                        config.num_random_future_samples
                        * (4 - cl_strategy.experience_step)
                    ),
                    mode=config.future_samples_mode,
                    config=config,
                )

                train_experience.dataset = train_experience.dataset + future_dataset

        # Train VQ-VAE
        cl_strategy.train(train_experience, [test_experience])

        # Train linear classifier, but before we freeze model params
        # We train two classifiers. One to predict all classes,
        # another to predict only observed so far classes.
        cl_strategy.model.freeze()

        # Train classifier
        print(f"Train classifier..")
        all_clf_head = train_classifier_on_all_classes(
            strategy=cl_strategy, config=config, benchmark=benchmark, device=device
        ).to(device)
        train_classifier_on_observed_only_classes(
            strategy=cl_strategy, config=config, benchmark=benchmark, device=device
        ).to(device)

        # cl_strategy.model.set_clf_head(all_clf_head)

        # Train new image gpt model
        print(f"Train igpt..")
        image_gpt = train_igpt(
            strategy=cl_strategy,
            config=config,
            train_dataset=igpt_train_dataset,
            device=device,
            n_layer=config.num_gpt_layers,
            classes_seen_so_far=train_experience.classes_seen_so_far,
        )

        # Evaluate VQ-VAE and linear classifier
        # cl_strategy.eval(benchmark.test_stream)

        # Reset linear classifier and unfreeze params
        # cl_strategy.model.reset_clf_head()

        cl_strategy.model.unfreeze()
        cl_strategy.experience_step += 1

    if is_using_wandb:
        log_summary_table_to_wandb(benchmark.train_stream, benchmark.test_stream)


def main(args):
    # turn on multiplication speed up in ampere series
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    # Reading configuration from ini file
    assert (
        args.config
    ), "Please fill the --config argument with valid path to configuration file."
    ini_config = ConfigParser()
    ini_config.read(args.config)

    # Unpack config to typed config class and create the model
    config = TrainConfig.construct_typed_config(ini_config)
    overwrite_config_with_args(args, config)

    # Fix num_class_embeddings in case of non-separated codebook
    # if not config.separate_codebooks:
    #     config.num_class_embeddings = config.num_embeddings

    # Construct wandb params if necessary
    is_using_wandb = (
        config.train_logger == "wandb"
        or config.evaluation_logger == "wandb"
        or args.run_id
    )
    if is_using_wandb:
        if args.dev:
            os.environ["WANDB_MODE"] = "offline"

        wandb_params = get_wandb_params(args, config)

        wandb.run.name = args.experiment_name or (
            f"BS-{config.batch_size * config.accumulate_grad_batches} | "
            f"#Emb-{config.num_embeddings} | "
            f"DEmb-{config.embedding_dim} | "
        )
        wandb_params["name"] = wandb.run.name
        wandb_params["id"] = wandb.run.id
    else:
        wandb_params = None

    # Fix path params
    today = datetime.datetime.now()
    run_id = wandb_params["id"] if wandb_params else today.strftime("%Y_%m_%d_%H_%M")

    config.checkpoint_path += f"/{run_id}/model"
    config.best_model_prefix += f"/{run_id}/best_model"
    config.bootstrapped_dataset_path += f"/{run_id}/bootstrapped_dataset"

    Path(config.checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(config.best_model_prefix).mkdir(parents=True, exist_ok=True)
    Path(config.bootstrapped_dataset_path).mkdir(parents=True, exist_ok=True)

    # Moving dataset to tmp
    datasets_dir = pathlib.Path(config.dataset_path)
    tmp = os.environ.get("TMPDIR", "/tmp")
    target_dataset_dir = pathlib.Path(f"{tmp}/dzverev_data/")
    target_dataset_dir.mkdir(exist_ok=True)

    zip_path = datasets_dir / "cifar-10-python.tar.gz"
    dataset_path = datasets_dir / "cifar-10-batches-py"

    target_zip_path = target_dataset_dir / "cifar-10-python.tar.gz"
    target_dataset_path = target_dataset_dir / "cifar-10-batches-py"

    if zip_path.exists() and not target_zip_path.exists():
        shutil.copy(str(zip_path), str(target_zip_path))

    if dataset_path.exists() and not target_dataset_path.exists():
        shutil.copytree(str(dataset_path), str(target_dataset_path))

    # Create benchmark
    benchmark = get_benchmark(config, target_dataset_dir)

    device = get_device(config)
    model = get_model(config, device)

    # Create evaluation plugin and train/val loggers
    cl_strategy_logger, eval_plugin_loggers = get_loggers(config, model, wandb_params)
    evaluation_plugin = get_evaluation_plugin(
        benchmark, eval_plugin_loggers, is_using_wandb
    )

    cl_strategy = NaivePytorchLightning(
        precision=config.precision,
        accelerator=config.accelerator,
        devices=config.devices,
        validate_every_n=config.validate_every_n,
        accumulate_grad_batches=config.accumulate_grad_batches,
        train_logger=cl_strategy_logger,
        initial_resume_from=args.resume_from,
        model=model,
        device=device,
        optimizer=model.configure_optimizers()["optimizer"],
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
        # plugins=[ReconstructionVisualizationPlugin(num_tasks_in_batch=2)],
        train_plugins=get_train_plugins(config),
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
