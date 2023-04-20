import torch
from pytorch_lightning import Callback
from torchvision.transforms import Compose, Normalize

from avalanche.evaluation.metrics import timing_metrics
from avalanche.training.plugins import EvaluationPlugin
from src.rnd.callbacks.log_generated_images import LogSampledImagesCallback
from src.rnd.configuration.config import TrainConfig
from src.rnd.metrics.rnd_accuracy import rnd_accuracy_metrics
from src.rnd.metrics.rnd_confusion_matrix import rnd_confusion_matrix_metrics
from src.rnd.metrics.rnd_forgetting import rnd_forgetting_metrics
from src.rnd.metrics.rnd_loss import rnd_loss_metrics
from src.rnd.model.gan_generator import MNISTGanGenerator
from src.rnd.model.rnd import RND
from src.rnd.model.vae_generator import MNISTVaeLinearGenerator

_default_mnist_train_transform = Compose([Normalize((0.1307,), (0.3081,))])

import typing as t


def get_evaluation_plugin(
    benchmark, evaluation_loggers, is_using_wandb
) -> t.Optional[EvaluationPlugin]:
    eval_plugin = EvaluationPlugin(
        timing_metrics(epoch_running=True),
        rnd_forgetting_metrics(experience=True, stream=True, accuracy=True),
        rnd_accuracy_metrics(experience=True, stream=True),
        rnd_confusion_matrix_metrics(
            num_classes=benchmark.n_classes,
            save_image=False,
            stream=True,
            wandb=is_using_wandb,
        ),
        rnd_loss_metrics(experience=True, stream=True),
        suppress_warnings=True,
        loggers=evaluation_loggers,
    )

    return eval_plugin


def get_model(config: TrainConfig, device: torch.device) -> RND:
    assert config.generator_checkpoint, "Generator checkpoint is necessary to provide!"
    if config.generator_type == "gan":
        generator = MNISTGanGenerator(input_dim=config.input_dim, output_dim=784)
    elif config.generator_type == "vae":
        generator = MNISTVaeLinearGenerator(
            x_dim=784,
            h_dim1=512,
            h_dim2=256,
            z_dim=2,
            transforms=_default_mnist_train_transform,
        )
    else:
        raise AssertionError("Unknown type of generator!")

    generator.load_state_dict(
        torch.load(config.generator_checkpoint, map_location=torch.device("cpu"))
    )
    generator.to(device)

    model = RND(
        generator=generator,
        num_random_images=config.num_random_images,
        l2_threshold=config.l2_threshold,
        rnd_latent_dim=config.rnd_latent_dim,
        num_classes=10,
        num_generation_attempts=config.num_generation_attempts,
    )

    return model


def get_callbacks() -> t.Callable[[int], t.List[Callback]]:
    return lambda x: [
        LogSampledImagesCallback(num_images=10),
    ]
