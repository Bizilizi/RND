import datetime
import os
import pathlib
import shutil
from configparser import ConfigParser
from pathlib import Path

import torch

import wandb
from avalanche.benchmarks import SplitCIFAR10
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from src.avalanche.strategies import NaivePytorchLightning
from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.data.transformations import (
    cifar10_to_tensor_and_normalization,
    cifar100_to_tensor_and_normalization,
    imagenet_to_tensor_and_normalization,
    cifar_augmentations,
    imagenet_augmentations,
)
from src.transformer_vq_vae.init_scrips import (
    get_benchmark,
    get_callbacks,
    get_evaluation_plugin,
    get_model,
    get_train_plugins,
)
from src.transformer_vq_vae.model_future import model_future_samples
from src.transformer_vq_vae.train_classifier import (
    train_classifier_on_all_classes,
    train_classifier_on_observed_only_classes,
)
from src.transformer_vq_vae.train_image_gpt import bootstrap_past_samples, train_igpt
from src.transformer_vq_vae.utils.copy_dataset_to_tmp import copy_dataset_to_tmp
from src.transformer_vq_vae.utils.wrap_empty_indices import (
    convert_avalanche_dataset_to_vq_mae_dataset,
)
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config_with_args
from train_utils import get_device, get_loggers, get_wandb_params


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


def reset_transformations_for_igpt(config: TrainConfig, dataset: ClassificationDataset):
    if config.dataset == "cifar10":
        return dataset.replace_current_transform_group(
            cifar10_to_tensor_and_normalization
        )
    elif config.dataset == "cifar100":
        return dataset.replace_current_transform_group(
            cifar100_to_tensor_and_normalization
        )
    elif config.dataset in ["tiny-imagenet", "imagenet"]:
        return dataset.replace_current_transform_group(
            imagenet_to_tensor_and_normalization
        )


def apply_transformations_for_bootstrapped_dataset(
    config: TrainConfig, dataset: ClassificationDataset
):
    if config.dataset == "cifar10":
        return dataset.replace_current_transform_group(cifar_augmentations)
    elif config.dataset == "cifar100":
        return dataset.replace_current_transform_group(cifar_augmentations)
    elif config.dataset in ["tiny-imagenet", "imagenet"]:
        return dataset.replace_current_transform_group(imagenet_augmentations)


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

        # IGPT data set is:
        #  - original dataset but with validation transformation
        #  - then wrapped to a MAE dataset format
        igpt_train_dataset = reset_transformations_for_igpt(
            config, train_experience.dataset
        )
        igpt_train_dataset = convert_avalanche_dataset_to_vq_mae_dataset(
            igpt_train_dataset,
            num_neighbours=config.quantize_top_k,
            config=config,
            time_tag=0,
        )

        # MAE dataset is original dataset but wrapped with extra fields
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

        # Bootstrap old data and modeled future samples
        if cl_strategy.experience_step != 0 and image_gpt is not None:
            image_gpt.to(device)
            cl_strategy.model.to(device)

            if config.num_random_past_samples != 0:
                print(f"Bootstrap vae model..")
                previous_classes = list(
                    set(train_experience.classes_seen_so_far).difference(
                        train_experience.classes_in_this_experience
                    )
                )
                (
                    mae_bootstrapped_dataset,
                    igpt_bootstrapped_dataset,
                ) = bootstrap_past_samples(
                    image_gpt=image_gpt,
                    vq_vae_model=cl_strategy.model,
                    num_images=get_num_random_past_samples(config, cl_strategy),
                    dataset_path=config.bootstrapped_dataset_path,
                    config=config,
                    experience_step=cl_strategy.experience_step,
                    classes_seen_so_far=previous_classes,
                )

                # IMPORTANT: Training dataset is augmented dataset + augmented bootstrapped dataset
                train_experience.dataset = (
                    train_experience.dataset + mae_bootstrapped_dataset
                )
                # IMPORTANT: IGPT dataset is un-augmented dataset + un-augmented bootstrapped dataset
                igpt_train_dataset = igpt_train_dataset + igpt_bootstrapped_dataset

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
        train_classifier_on_all_classes(
            strategy=cl_strategy, config=config, benchmark=benchmark, device=device
        ).to(device)
        train_classifier_on_observed_only_classes(
            strategy=cl_strategy, config=config, benchmark=benchmark, device=device
        ).to(device)

        # Train new image gpt model
        print(f"Train igpt..")
        image_gpt = train_igpt(
            strategy=cl_strategy,
            config=config,
            train_dataset=igpt_train_dataset,
            device=device,
            n_layer=config.num_gpt_layers,
            classes_seen_so_far=train_experience.classes_seen_so_far,
            num_all_classes=benchmark.n_classes,
            epoch_num=config.igpt_epoch_num,
        )

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

        wandb.run.summary["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", -1)
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
    tmp = os.environ.get("TMPDIR", "/tmp")
    target_dataset_dir = pathlib.Path(f"{tmp}/dzverev_data/{config.dataset}")
    target_dataset_dir.mkdir(exist_ok=True, parents=True)

    copy_dataset_to_tmp(config, target_dataset_dir)

    # Create benchmark
    benchmark = get_benchmark(config, target_dataset_dir)

    device = get_device(config)
    model = get_model(config, device, benchmark)

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
