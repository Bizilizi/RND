import datetime
from configparser import ConfigParser

import torch
from torchvision import transforms

import wandb
from avalanche.benchmarks import SplitCIFAR10
from src.avalanche.strategies import NaivePytorchLightning
from src.transformer_vq_vae.callbacks.reconstruction_visualization_plugin import (
    ReconstructionVisualizationPlugin,
)
from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.init_scrips import (
    get_callbacks,
    get_evaluation_plugin,
    get_model,
)
from src.transformer_vq_vae.model_future import model_future_samples
from src.transformer_vq_vae.train_classifier import (
    train_classifier_on_all_classes,
    train_classifier_on_observed_only_classes,
)
from src.transformer_vq_vae.train_image_gpt import bootstrap_past_samples, train_igpt
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config_with_args
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

        # bootstrap old data and modeled future samples
        if cl_strategy.experience_step != 0 and image_gpt is not None:
            image_gpt.to(device)
            cl_strategy.model.to(device)

            if config.num_random_past_samples != 0:
                print(f"Bootstrap vae model..")
                bootstrapped_dataset = bootstrap_past_samples(
                    image_gpt=image_gpt,
                    vq_vae_model=cl_strategy.model,
                    num_images=(
                        config.num_random_past_samples * cl_strategy.experience_step
                    ),
                    dataset_path=config.bootstrapped_dataset_path,
                    config=config,
                    sos_token=config.num_embeddings + 1,
                    experience_step=cl_strategy.experience_step,
                )

                train_experience.dataset = (
                    train_experience.dataset
                    + bootstrapped_dataset.replace_current_transform_group(
                        transforms.Compose(
                            [
                                transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                            ]
                        )
                    )
                )

                train_dataset = train_dataset + bootstrapped_dataset

            if config.num_random_future_samples != 0:
                print(f"Model future samples..")
                future_dataset = model_future_samples(
                    vq_vae_model=cl_strategy.model,
                    num_rand_samples=(
                        config.num_random_future_samples
                        * (4 - cl_strategy.experience_step)
                    ),
                    mode=config.future_samples_mode,
                )

                train_experience.dataset = train_experience.dataset + future_dataset

        # Train VQ-VAE
        cl_strategy.train(train_experience, [test_experience])

        # Train linear classifier, but before we freeze model params
        # We train two classifiers. One to predict all classes,
        # another to predict only observed so far classes.
        cl_strategy.model.freeze()

        # Train new image gpt model
        print(f"Train igpt..")
        image_gpt = train_igpt(
            strategy=cl_strategy,
            config=config,
            train_dataset=train_dataset + val_dataset,
            device=device,
            sos_token=config.num_embeddings + 1,
            mask_token=config.num_embeddings,
            n_layer=config.num_gpt_layers,
        )

        # Train classifier
        print(f"Train classifier..")
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
    config.bootstrapped_dataset_path += f"/{run_id}/best_model"

    benchmark = SplitCIFAR10(
        n_experiences=5,
        return_task_id=True,
        shuffle=True,
        dataset_root=config.dataset_path,
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
        optimizer=model.configure_optimizers()[0][0],
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
