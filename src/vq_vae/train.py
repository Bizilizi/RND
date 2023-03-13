import pathlib
from configparser import ConfigParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torchvision import transforms
import shutil
import wandb
from avalanche.benchmarks import SplitCIFAR10
from torchvision import datasets

from src.avalanche.data import PLDataModule
from src.avalanche.strategies import NaivePytorchLightning
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config_with_args
from src.vq_vae.callbacks.reconstruction_visualization_plugin import (
    ReconstructionVisualizationPlugin,
)
from src.vq_vae.configuration.config import TrainConfig
from src.vq_vae.init_scrips import get_evaluation_plugin, get_callbacks, get_model
from src.vq_vae.model.classification_head import CnnClassifier
from src.vq_vae.model.vq_vae import VQVae
from train_utils import (
    get_device,
    get_loggers,
    get_wandb_params,
)


def train_classifier(
    strategy: NaivePytorchLightning,
    config: TrainConfig,
    benchmark: SplitCIFAR10,
):
    model = strategy.model
    clf_head = CnnClassifier(
        in_channels=config.embedding_dim,
        num_classes=benchmark.n_classes,
        vq_vae=model,
        experience_step=strategy.experience_step,
    )

    train_dataset = datasets.CIFAR10(
        root=config.dataset_path,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
            ]
        ),
    )
    test_dataset = datasets.CIFAR10(
        root=config.dataset_path,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
            ]
        ),
    )
    datamodule = PLDataModule(
        batch_size=256,
        num_workers=config.num_workers,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )

    # Training
    trainer = Trainer(
        check_val_every_n_epoch=strategy.validate_every_n,
        accelerator=strategy.accelerator,
        devices=strategy.devices,
        logger=strategy.train_logger,
        callbacks=[
            EarlyStopping(
                monitor=f"val/clf_accuracy/experience_step_{strategy.experience_step}",
                mode="max",
                patience=10,
            )
        ],
        max_epochs=2,
    )

    trainer.fit(clf_head, datamodule=datamodule)

    return clf_head


def train_loop(
    benchmark: SplitCIFAR10,
    cl_strategy: NaivePytorchLightning,
    is_using_wandb: bool,
    config: TrainConfig,
) -> None:
    """
    :return:
    """

    for train_experience, test_experience in zip(
        benchmark.train_stream, benchmark.test_stream
    ):
        # Train VQ-VAE and linear classifier
        cl_strategy.train(train_experience, [test_experience])
        cl_strategy.model.set_clf_head(
            train_classifier(strategy=cl_strategy, config=config, benchmark=benchmark)
        )

        # Evaluate VQ-VAE and linear classifier
        cl_strategy.eval(benchmark.test_stream)
        cl_strategy.model.reset_clf_head()

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
            f"RI-0."
            f"RN-{config.num_random_noise}."
            f"Dr-{config.regularization_dropout}."
            f"Wd-{config.regularization_lambda}."
        )
        wandb_params["name"] = wandb.run.name
    else:
        wandb_params = None

    # Create benchmark, model and loggers
    datasets_dir = pathlib.Path(config.dataset_path)
    target_dataset_dir = pathlib.Path("/tmp/dzverev_data/")
    target_dataset_dir.mkdir(exist_ok=True)

    zip_path = datasets_dir / "cifar-10-python.tar.gz"
    dataset_path = datasets_dir / "cifar-10-batches-py"

    target_zip_path = target_dataset_dir / "cifar-10-python.tar.gz"
    target_dataset_path = target_dataset_dir / "cifar-10-batches-py"

    if zip_path.exists() and not target_zip_path.exists():
        shutil.copy(str(zip_path), str(target_zip_path))

    if dataset_path.exists() and not target_dataset_path.exists():
        shutil.copytree(str(dataset_path), str(target_dataset_path))

    benchmark = SplitCIFAR10(
        n_experiences=5,
        return_task_id=True,
        shuffle=False,
        dataset_root="/tmp/dzverev_data",
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
        plugins=[ReconstructionVisualizationPlugin()],
    )

    # Run training process
    print(f"Running training process..")
    try:
        train_loop(
            benchmark=benchmark,
            cl_strategy=cl_strategy,
            is_using_wandb=is_using_wandb,
            config=config,
        )
    except KeyboardInterrupt:
        print("Training successfully interrupted.")
