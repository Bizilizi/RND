import typing as t

import pytorch_lightning as pl
import torch
import torch.nn as nn
from avalanche.benchmarks import CLExperience
from avalanche.training.templates.base import ExpSequence
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torchvision import models

from src.model.rnd.generator import ImageGenerator


class RND(pl.LightningModule):
    experience_step: int

    def __init__(
        self,
        generator: ImageGenerator,
        num_random_images: int,
        l2_threshold: float,
        *,
        image_generation_batch_size: int = 10,
        rnd_latent_dim: int = 200,
        num_classes: int = 100,
        num_generation_attempts: int = 20,
    ):
        super().__init__()
        self.rnd_latent_dim = rnd_latent_dim
        self.num_classes = num_classes
        self.generator = generator
        self.num_random_images = num_random_images
        self.l2_threshold = l2_threshold
        self.num_generation_attempts = num_generation_attempts
        self.image_generation_batch_size = image_generation_batch_size

        # Random network
        self.random_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, self.rnd_latent_dim),
        )
        for param in self.random_network.parameters():
            param.requires_grad = False

        # Learnable network that predicts outputs of random network
        # First rnd_latent_dim elements of predicted tensor will be compared to
        # random network output. The rest will go to downstream head to create
        # classification output.
        self.module = nn.Sequential(
            nn.Conv2d(1, 3, 3),
            models.resnet18(),
        )
        self.downstream_head = nn.Sequential(
            nn.Linear(1000 - self.rnd_latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, self.num_classes),
            nn.ReLU(),
        )

        # Create losses
        self.rnd_loss = nn.MSELoss()
        self.downstream_loss = nn.CrossEntropyLoss()

        # Flags
        self.keep_logging = True  # Log data and losses to logger
        self.keep_sampling = True  # Sample random images and add them to the batch

    def _generate_random_images_with_low_l2(self) -> t.Optional[torch.Tensor]:
        samples = []
        image_generation_attempts = 0

        with torch.no_grad():
            for _ in range(self.num_generation_attempts):
                if len(samples) >= self.num_random_images:
                    break

                for _ in range(
                    self.num_random_images
                    * self.num_classes
                    // self.image_generation_batch_size
                ):
                    random_x = self.generator.generate(
                        self.image_generation_batch_size, device=self.device
                    )

                    # Perform forward step on randomly generated data
                    random_rn_target = self.random_network(random_x)
                    random_module_output = self.module(random_x)

                    random_module_rn_pred = random_module_output[
                        :, : self.rnd_latent_dim
                    ]

                    # Compute mask based on given l2 threshold
                    # then we apply it to network prediction and targets for random data
                    threshold_mask = (random_module_rn_pred - random_rn_target).pow(
                        2
                    ).mean(dim=1) < self.l2_threshold

                    if threshold_mask.any():
                        samples.append(random_x[threshold_mask])

                image_generation_attempts += 1

        samples = samples[: self.num_random_images]

        if self.keep_logging:
            self.log_with_postfix(
                f"image_generation_attempts",
                image_generation_attempts,
                on_epoch=True,
            )
            self.log_with_postfix(
                f"num_incomplete_generation_attempts",
                1 if len(samples) < self.num_random_images else 0,
                on_epoch=True,
                reduce_fx="sum",
            )
            self.log_with_postfix(
                f"generated_samples",
                len(samples),
                on_epoch=True,
            )

        if samples:
            return torch.cat(samples)

        return None

    def forward(
        self, x: torch.Tensor, task_label: t.Optional[torch.Tensor] = None
    ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rn_target = self.random_network(x)
        module_output = self.module(x)

        module_rn_pred = module_output[:, : self.rnd_latent_dim]
        module_downstream_pred = module_output[:, self.rnd_latent_dim :]
        module_downstream_pred = self.downstream_head(module_downstream_pred)

        return rn_target, module_rn_pred, module_downstream_pred

    def criterion(
        self,
        x: t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        y: t.Optional[torch.Tensor] = None,
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:

        """
        module_output is a tensor with dim = (batch_size, 1000)
        """
        rn_target, module_rn_pred, module_downstream_pred = x

        # Compute losses
        rnd_loss = self.rnd_loss(module_rn_pred, rn_target)

        if y is not None:
            downstream_loss = self.downstream_loss(module_downstream_pred, y)
        else:
            downstream_loss = self.downstream_loss(
                module_downstream_pred,
                module_downstream_pred.argmax(dim=1),
            )

        return rnd_loss, downstream_loss

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch

        # Compute losses
        x = self.forward(x)
        rnd_loss, downstream_loss = self.criterion(x, y)

        # Perform forward step on randomly generated data if necessary
        if self.keep_sampling:
            random_x = self._generate_random_images_with_low_l2()

            if random_x is not None:
                random_x = self.forward(random_x)
                random_rnd_loss, random_downstream_loss = self.criterion(random_x)

                rnd_loss += random_rnd_loss
                downstream_loss += random_downstream_loss

        loss = rnd_loss + downstream_loss
        if self.keep_logging:
            self.log_with_postfix(
                f"train/rnd_loss",
                rnd_loss,
            )
            self.log_with_postfix(
                f"train/downstream_loss",
                downstream_loss,
            )
            self.log_with_postfix(f"train/loss", loss)

        return {"loss": loss, "forward_output": x}

    def validation_step(self, batch, batch_idx):
        x, y, *_ = batch

        # Compute losses
        x = self.forward(x)
        rnd_loss, downstream_loss = self.criterion(x, y)
        loss = rnd_loss + downstream_loss

        if self.keep_logging:
            self.log_with_postfix("val/rnd_loss", rnd_loss)
            self.log_with_postfix("val/downstream_loss", downstream_loss)
            self.log_with_postfix("val/loss", loss)

        return {"loss": loss, "forward_output": x}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-3)
        return optimizer

    def log_with_postfix(self, name: str, value: t.Any, *args, **kwargs):
        self.log_dict(
            {
                f"{name}/experience_step_{self.experience_step}": value,
            },
            *args,
            **kwargs,
        )
