import datetime
import os
import pathlib
import shutil
from configparser import ConfigParser
import typing as t
import torch
from torchvision import transforms

import wandb
from avalanche.benchmarks import SplitCIFAR10, SplitCIFAR100
from src.avalanche.strategies import NaivePytorchLightning
from src.transformer_vq_vae.callbacks.reconstruction_visualization_plugin import (
    ReconstructionVisualizationPlugin,
)
from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.init_scrips import (
    get_callbacks,
    get_evaluation_plugin,
    get_model,
    get_train_plugins,
    get_benchmark,
)
from src.transformer_vq_vae.model_future import model_future_samples
from src.transformer_vq_vae.train_classifier import (
    train_classifier_on_all_classes,
    train_classifier_on_observed_only_classes,
)
from src.transformer_vq_vae.train_image_gpt import bootstrap_past_samples, train_igpt
from src.transformer_vq_vae.utils.wrap_empty_indices import (
    wrap_dataset_with_empty_indices,
)
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config_with_args
from train_utils import get_device, get_loggers, get_wandb_params

from pathlib import Path


def get_epochs_schedule(config: TrainConfig):
    if config.num_epochs_schedule == "fixed":
        return config.max_epochs

    if config.num_epochs_schedule == "schedule":
        schedule = torch.linspace(
            config.max_epochs, config.min_epochs, config.num_tasks
        )
        return schedule.tolist()

    if config.num_epochs_schedule == "warmup":
        schedule = [config.min_epochs] * config.num_tasks
        schedule[0] = config.max_epochs

        return schedule


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
    resume_arguments: t.Optional[t.Dict[str, t.Any]],
) -> None:
    """
    :return:
    """

    image_gpt = None
    if resume_arguments:
        start_experience = resume_arguments["current_experience_step"]
        print(f"Resume experience step from: {start_experience}")
    else:
        start_experience = 0

    for train_experience, test_experience in zip(
        benchmark.train_stream[start_experience:],
        benchmark.test_stream[start_experience:],
    ):
        train_experience.dataset = wrap_dataset_with_empty_indices(
            train_experience.dataset, num_neighbours=config.quantize_top_k
        )
        test_experience.dataset = wrap_dataset_with_empty_indices(
            test_experience.dataset, num_neighbours=config.quantize_top_k
        )
        igpt_train_dataset = train_experience.dataset + test_experience.dataset

        # Bootstrap old data and modeled future samples
        if cl_strategy.experience_step != 0 and image_gpt is not None:
            image_gpt.to(device)
            cl_strategy.model.to(device)

            # Restore model params from previous CL step
            # Only in case this run was restored according to resume_arguments
            if resume_arguments and resume_arguments["current_experience_step"] > 0:
                experience_step = cl_strategy.experience_step - 1
                chkp_path = resume_arguments[
                    f"experience_step_{experience_step}/model_checkpoint_path"
                ]
                state_dict = torch.load(chkp_path)["state_dict"]
                cl_strategy.model.load_state_dict(state_dict)

            if config.num_random_past_samples != 0:
                print(f"Bootstrap vae model..")
                previous_classes = list(
                    set(train_experience.classes_seen_so_far).difference(
                        train_experience.classes_in_this_experience
                    )
                )

                bootstrapped_dataset = bootstrap_past_samples(
                    image_gpt=image_gpt,
                    vq_vae_model=cl_strategy.model,
                    num_images=get_num_random_past_samples(config, cl_strategy),
                    dataset_path=config.bootstrapped_dataset_path,
                    config=config,
                    experience_step=cl_strategy.experience_step,
                    classes_seen_so_far=previous_classes,
                    num_classes=benchmark.n_classes,
                )

                train_experience.dataset = (
                    train_experience.dataset + bootstrapped_dataset
                )

                igpt_train_dataset = igpt_train_dataset + bootstrapped_dataset

        # Train VQ-VAE
        checkpoint_fully_trained = resume_arguments and (
            resume_arguments["current_epochs"] < resume_arguments["current_max_epochs"]
        )
        if not checkpoint_fully_trained:
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
            image_gpt=image_gpt if config.reuse_igpt else None,
            classes_seen_so_far=train_experience.classes_seen_so_far,
            num_classes=benchmark.n_classes,
        )

        # Unfreeze model and move to the next experience
        cl_strategy.model.unfreeze()
        cl_strategy.experience_step += 1

        # If we resume, we resume only once
        resume_arguments = False

    if is_using_wandb:
        log_summary_table_to_wandb(benchmark.train_stream, benchmark.test_stream)


def main(args):
    resume_arguments = torch.load(args.resume_from) if args.resume_from else None

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
    if not config.separate_codebooks:
        config.num_class_embeddings = config.num_embeddings

    # Construct wandb params if necessary
    is_using_wandb = (
        config.train_logger == "wandb"
        or config.evaluation_logger == "wandb"
        or args.run_id
    )
    if is_using_wandb:
        if args.dev:
            os.environ["WANDB_MODE"] = "offline"

        wandb_params = get_wandb_params(args, config, resume_arguments)

        wandb.run.name = args.experiment_name or (
            f"BS-{config.batch_size * config.accumulate_grad_batches} | "
            f"#Emb-{config.num_embeddings} | "
            f"DEmb-{config.embedding_dim} | "
        )
        wandb_params["name"] = wandb.run.name
        wandb_params["id"] = wandb.run.id

        wandb.run.summary["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", -1)
        if resume_arguments:
            wandb.run.summary["restored_from"] = resume_arguments[
                "current_model_checkpoint_path"
            ]
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
    model = get_model(config, device, benchmark)

    # Create evaluation plugin and train/val loggers
    cl_strategy_logger, eval_plugin_loggers = get_loggers(config, model, wandb_params)
    evaluation_plugin = get_evaluation_plugin(
        benchmark, eval_plugin_loggers, is_using_wandb
    )

    epochs_schedule = get_epochs_schedule(config)
    cl_strategy = NaivePytorchLightning(
        precision=config.precision,
        accelerator=config.accelerator,
        devices=config.devices,
        validate_every_n=config.validate_every_n,
        accumulate_grad_batches=config.accumulate_grad_batches,
        train_logger=cl_strategy_logger,
        model=model,
        device=device,
        strategy=config.strategy,
        optimizer=model.configure_optimizers()["optimizer"],
        criterion=model.criterion,
        train_mb_size=config.batch_size,
        train_mb_num_workers=config.num_workers,
        train_epochs=config.max_epochs,
        eval_mb_size=config.batch_size,
        evaluator=evaluation_plugin,
        callbacks=get_callbacks(config),
        max_epochs=epochs_schedule,
        min_epochs=epochs_schedule,
        best_model_path_prefix=config.best_model_prefix,
        plugins=[ReconstructionVisualizationPlugin(num_tasks_in_batch=2)],
        train_plugins=get_train_plugins(config),
        resume_arguments=resume_arguments,
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
            resume_arguments=resume_arguments,
        )
    except KeyboardInterrupt:
        print("Training successfully interrupted.")
