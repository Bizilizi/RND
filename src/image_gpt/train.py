import datetime
import os
import pathlib
from configparser import ConfigParser

import torch
from torch import distributed

import wandb
from avalanche.benchmarks import SplitCIFAR10
from torch.utils.data import ConcatDataset, Dataset

from src.avalanche.strategies import NaivePytorchLightning
from src.image_gpt.configuration.config import TrainConfig
from src.image_gpt.init_scrips import (
    get_model,
    get_benchmark,
    get_cl_strategy,
)
from src.image_gpt.train_diffusion import (
    bootstrap_past_samples as bootstrap_past_samples_with_diffusion,
    train_diffusion,
)
from src.image_gpt.train_image_gpt import (
    bootstrap_past_samples as bootstrap_past_samples_with_igpt,
    train_igpt,
)
from src.image_gpt.utils.copy_dataset import copy_dataset_to_tmp
from src.image_gpt.utils.fid_score import calculate_fid_given_datasets
from src.image_gpt.utils.wrap_empty_indices import (
    wrap_dataset,
)
from src.utils.train_script import overwrite_config_with_args
from train_utils import get_device, get_wandb_params

from pathlib import Path


class ImgDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, *_ = self.dataset[item]
        return x["images"]


def train_loop(
    benchmark: SplitCIFAR10,
    cl_strategy: NaivePytorchLightning,
    config: TrainConfig,
    device: torch.device,
    is_using_wandb: bool,
    is_distributed: bool,
    local_rank: int,
    resume_from: str,
) -> None:
    """
    :return:
    """

    cl_strategy.model.load_state_dict(
        torch.load(
            resume_from,
            map_location=torch.device("cpu"),
        )["state_dict"]
    )
    cl_strategy.model.to(device)

    print(f"Train igpt..")
    igpt_train_dataset = ConcatDataset(
        [
            wrap_dataset(
                experience.dataset,
                img_embedding_dim=config.img_embedding_dim,
                is_past_domain=True,
            )
            for experience in benchmark.train_stream
        ]
    )
    if config.sampler_type == 'igpt':
        image_gpt = train_igpt(
            strategy=cl_strategy,
            config=config,
            train_dataset=igpt_train_dataset,
            device=device,
            local_rank=local_rank,
            is_distributed=is_distributed,
            num_classes=benchmark.n_classes,
            classes_seen_so_far=range(10),
        )

        bootstrapped_dataset = bootstrap_past_samples_with_igpt(
            image_gpt=image_gpt,
            qmae_model=cl_strategy.model,
            num_images=25000,
            config=config,
            classes_seen_in_past=range(10),
        )
    elif config.sampler_type == 'diffusion':
        diffusion = train_diffusion(
            strategy=cl_strategy,
            config=config,
            train_dataset=igpt_train_dataset,
            device=device,
            local_rank=local_rank,
            is_distributed=is_distributed,
            num_classes=benchmark.n_classes,
            classes_seen_so_far=range(10),
        )

        bootstrapped_dataset = bootstrap_past_samples_with_diffusion(
            diffusion=diffusion,
            qmae_model=cl_strategy.model,
            num_images=25000,
            config=config,
            classes_seen_in_past=range(10),
        )
    else:
        assert False, f"wrong sampler type -- {config.sampler_type}"

    fid_score = calculate_fid_given_datasets(
        ImgDataset(igpt_train_dataset),
        ImgDataset(bootstrapped_dataset),
        128,
        device,
        2048,
    )

    if is_using_wandb:
        wandb.log({"fid_score/all_tasks": fid_score})
    else:
        print(f"Fid score: {fid_score}")


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

    if is_main_process:

        # this will restore wand run from args id
        with wandb.init(
            project=args.project,
            id=args.run_id,
            entity="vgg-continual-learning",
            group=args.group,
            dir=args.wandb_dir,
            resume="must",
        ) as qmae_run:
            for k, v in qmae_run.config.items():
                if k == "accelerator" or "gpt" in k:
                    continue
                setattr(config, k, v)

        if args.dev:
            os.environ["WANDB_MODE"] = "offline"
        # Now we will re-init wandb with igpt project
        wandb_params = dict(
            project=args.model.lower(),
            entity="vgg-continual-learning",
            group=args.group,
            dir=args.wandb_dir,
            reinit=False,
        )
        wandb.init(**wandb_params)

        wandb.run.name = args.experiment_name or (
            f"BS-{config.batch_size * config.accumulate_grad_batches} | "
            f"#Emb-{config.num_embeddings} | "
            f"DEmb-{config.embedding_dim} | "
        )

        wandb.config.update(dict(config))
        wandb_params["config"] = config

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
        is_using_wandb=True,
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
            is_using_wandb=True,
            is_distributed=is_distributed,
            local_rank=args.local_rank,
            resume_from=args.resume_from,
        )
    except KeyboardInterrupt:
        print("Training successfully interrupted.")

    distributed.destroy_process_group()
