import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import models

from src.model.rnd.generator import ImageGenerator


class RND(pl.LightningModule):
    def __init__(
        self,
        generator: ImageGenerator,
        num_random_images: int,
        l2_threshold: float,
        *,
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
            models.resnet152(pretrained=False),
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

    def _generate_random_images_with_low_l2(self):
        samples = []
        image_generation_attempts = 0

        with torch.no_grad():
            for _ in range(self.num_generation_attempts):
                if len(samples) >= self.num_random_images:
                    break

                random_x = self.generator.generate(
                    self.num_random_images * self.num_classes, device=self.device
                )

                # Perform forward step on randomly generated data
                random_rn_target = self.random_network(random_x)
                random_module_output = self.module(random_x)

                random_module_rn_pred = random_module_output[:, : self.rnd_latent_dim]
                random_module_downstream_pred = random_module_output[
                    :, self.rnd_latent_dim :
                ]

                # Compute mask based on given l2 threshold
                # then we apply it to network prediction and targets for random data
                threshold_mask = (
                    torch.pow(random_module_rn_pred - random_rn_target, 2).sum(dim=1)
                    < self.l2_threshold
                )

                if threshold_mask.any():
                    samples.append(random_x[threshold_mask])

                image_generation_attempts += 1

        samples = samples[: self.num_random_images]

        if self.keep_logging:
            self.log(
                "image_generation_attempts",
                image_generation_attempts,
                on_epoch=True,
            )
            self.log(
                "num_incomplete_generation_attempts",
                1 if len(samples) < self.num_random_images else 0,
                on_epoch=True,
                reduce_fx="sum",
            )
            self.log(
                "generated_samples",
                len(samples),
                on_epoch=True,
            )

        return torch.cat(samples)

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch

        # Perform forward step on the data from the batch
        batch_rn_target = self.random_network(x)
        batch_module_output = self.module(x)
        """
        batch_module_output is a tensor with dim = (batch_size, 1000)
        """
        batch_module_rn_pred = batch_module_output[:, : self.rnd_latent_dim]
        batch_module_downstream_pred = batch_module_output[:, self.rnd_latent_dim :]

        # Compute losses
        rnd_loss = self.rnd_loss(batch_module_rn_pred, batch_rn_target)
        downstream_loss = self.downstream_loss(batch_module_downstream_pred, y)

        # Perform forward step on randomly generated data if necessary
        if self.keep_sampling:
            random_x = self._generate_random_images_with_low_l2()

            random_rn_target = self.random_network(random_x)
            random_module_output = self.module(random_x)

            random_module_rn_pred = random_module_output[:, : self.rnd_latent_dim]
            random_module_downstream_pred = random_module_output[
                :, self.rnd_latent_dim :
            ]

            # Add losses from random data
            rnd_loss += self.rnd_loss(random_module_rn_pred, random_rn_target)
            downstream_loss += self.downstream_loss(
                random_module_downstream_pred,
                random_module_downstream_pred.argmax(dim=1),
            )

        loss = rnd_loss + downstream_loss

        if self.keep_logging:
            self.log("rnd_loss", rnd_loss, on_epoch=True)
            self.log("downstream_loss", downstream_loss, on_epoch=True)
            self.log("loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-3)
        return optimizer