import datetime
import os
import pathlib
from configparser import ConfigParser

import torch
from torch import distributed

import wandb
from avalanche.benchmarks import SplitCIFAR10
from src.avalanche.strategies import NaivePytorchLightning
from src.qmae_latent_extension.configuration.config import TrainConfig
from src.qmae_latent_extension.init_scrips import (
    get_model,
    get_benchmark,
    get_cl_strategy,
)
from src.qmae_latent_extension.train_classifier import (
    train_classifier_on_all_classes,
    train_classifier_on_observed_only_classes,
    train_classifier_on_random_memory,
)
from src.qmae_latent_extension.train_image_gpt import bootstrap_past_samples, train_igpt
from src.qmae_latent_extension.utils.copy_dataset import copy_dataset_to_tmp
from src.qmae_latent_extension.utils.gdumb import extend_memory
from src.qmae_latent_extension.utils.wrap_empty_indices import (
    wrap_dataset_with_empty_indices,
)
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config_with_args
from train_utils import get_device, get_wandb_params

from pathlib import Path


def get_num_random_past_samples(
    config: TrainConfig, cl_strategy: NaivePytorchLightning
):
    if config.num_random_past_samples_schedule == "fixed":
        return config.num_random_past_samples

    if config.num_random_past_samples_schedule == "linear":
        return config.num_random_past_samples * cl_strategy.experience_step

    if config.num_random_past_samples_schedule == "schedule":
        schedule = [0, 3000, 5000, 7000, 10000]
        return schedule[int(cl_strategy.experience_step)]


def train_loop(
    benchmark: SplitCIFAR10,
    cl_strategy: NaivePytorchLightning,
    config: TrainConfig,
    device: torch.device,
    is_using_wandb: bool,
    is_distributed: bool,
    local_rank: int,
) -> None:
    """
    :return:
    """
    if is_using_wandb and local_rank == 0:
        log_summary_table_to_wandb(benchmark.train_stream, benchmark.test_stream)

    image_gpt = None

    sos_token = cl_strategy.model.feature_quantization.num_embeddings + 1
    mask_token = cl_strategy.model.feature_quantization.num_embeddings

    for train_experience, test_experience in zip(
        benchmark.train_stream, benchmark.test_stream
    ):
        cl_strategy.model.calculate_class_maps(
            torch.tensor(train_experience.dataset.targets).unique()
        )

        train_experience.dataset = wrap_dataset_with_empty_indices(
            train_experience.dataset, time_index=cl_strategy.experience_step
        )
        test_experience.dataset = wrap_dataset_with_empty_indices(
            test_experience.dataset, time_index=cl_strategy.experience_step
        )
        igpt_train_dataset = train_experience.dataset

        # Bootstrap old data
        if (
            cl_strategy.experience_step != 0
            and image_gpt is not None
            and config.num_random_past_samples != 0
        ):
            print(f"Bootstrap vae model..")

            image_gpt.to(device)
            cl_strategy.model.to(device)

            bootstrapped_dataset = bootstrap_past_samples(
                image_gpt=image_gpt,
                vq_vae_model=cl_strategy.model,
                num_images=get_num_random_past_samples(config, cl_strategy),
                dataset_path=config.bootstrapped_dataset_path,
                config=config,
                sos_token=sos_token,
                experience_step=cl_strategy.experience_step,
                mask_token=mask_token,
            )

            train_experience.dataset = train_experience.dataset + bootstrapped_dataset
            igpt_train_dataset = igpt_train_dataset + bootstrapped_dataset

        # Train VQ-VAE
        if cl_strategy.experience_step > 0:
            cl_strategy.model.feature_quantization.extend_codebook()
            cl_strategy.model.extend_clf_head()

            sos_token = cl_strategy.model.feature_quantization.num_embeddings + 1
            mask_token = cl_strategy.model.feature_quantization.num_embeddings

        cl_strategy.train(train_experience, [test_experience])
        cl_strategy.model.freeze()

        # Train new image gpt model
        print(f"Train igpt..")
        image_gpt = train_igpt(
            strategy=cl_strategy,
            config=config,
            train_dataset=igpt_train_dataset,
            device=device,
            sos_token=sos_token,
            mask_token=mask_token,
            n_layer=config.num_gpt_layers,
            image_gpt=image_gpt,
            local_rank=local_rank,
            is_distributed=is_distributed,
        )

        # Train linear classifiers
        print(f"Train classifier..")
        # We train two classifiers. One to predict all classes,
        # another to predict only observed so far classes.
        train_classifier_on_all_classes(
            strategy=cl_strategy, config=config, benchmark=benchmark, device=device
        )
        train_classifier_on_observed_only_classes(
            strategy=cl_strategy, config=config, benchmark=benchmark, device=device
        )

        # Finish CL step
        cl_strategy.model.unfreeze()
        cl_strategy.experience_step += 1

        # Wait until the main process
        distributed.barrier()


def main(args):
    is_distributed = args.world_size > 1
    is_main_process = args.local_rank == 0

    # Init pytorch distributed
    distributed.init_process_group(
        init_method=f"tcp://localhost:{args.port}",
        world_size=args.world_size,
        rank=args.local_rank,
        group_name="cl_sync",
    )

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
    if is_using_wandb and is_main_process:
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

    # propagate run_id to other processes
    list_with_run_id = [None]

    if is_main_process:
        today = datetime.datetime.now()
        run_id = (
            wandb_params["id"] if wandb_params else today.strftime("%Y_%m_%d_%H_%M")
        )
        list_with_run_id = [run_id]

    distributed.broadcast_object_list(list_with_run_id)
    run_id = list_with_run_id[0]

    # Fix path params
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

    if is_main_process:
        copy_dataset_to_tmp(config, target_dataset_dir)

    # Wait until the dataset is loaded and unpacked
    distributed.barrier()

    # Create benchmark
    device = get_device(config, args.local_rank)

    benchmark = get_benchmark(config, target_dataset_dir)
    model = get_model(config, device)
    cl_strategy = get_cl_strategy(
        config=config,
        wandb_params=wandb_params,
        model=model,
        benchmark=benchmark,
        device=device,
        resume_from=args.resume_from,
        local_rank=args.local_rank,
        is_using_wandb=is_using_wandb,
        is_distributed=is_distributed,
    )

    # Run training process
    print(f"Running training process..")
    try:
        train_loop(
            benchmark=benchmark,
            cl_strategy=cl_strategy,
            config=config,
            device=device,
            is_using_wandb=is_using_wandb,
            is_distributed=is_distributed,
            local_rank=args.local_rank,
        )
    except KeyboardInterrupt:
        print("Training successfully interrupted.")

    distributed.destroy_process_group()
