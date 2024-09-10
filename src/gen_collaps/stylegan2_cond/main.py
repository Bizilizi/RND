"""
---
title: StyleGAN 2 Model Training
summary: >
 An annotated PyTorch implementation of StyleGAN2 model training code.
---

# [StyleGAN 2](index.html) Model Training

This is the training code for [StyleGAN 2](index.html) model.

![Generated Images](generated_64.png)

---*These are $64 \times 64$ images generated after training for about 80K steps.*---

*Our implementation is a minimalistic StyleGAN 2 model training code.
Only single GPU training is supported to keep the implementation simple.
We managed to shrink it to keep it at less than 500 lines of code, including the training loop.*

*Without DDP (distributed data parallel) and multi-gpu training it will not be possible to train the model
for large resolutions (128+).
If you want training code with fp16 and DDP take a look at
[lucidrains/stylegan2-pytorch](https://github.com/lucidrains/stylegan2-pytorch).*

We trained this on [CelebA-HQ dataset](https://github.com/tkarras/progressive_growing_of_gans).
You can find the download instruction in this
[discussion on fast.ai](https://forums.fast.ai/t/download-celeba-hq-dataset/45873/3).
Save the images inside [`data/stylegan` folder](#dataset_path).
"""

import argparse
import glob
import os
import pathlib
import re
import math
from pathlib import Path
from typing import Iterator, Tuple
import shutil

import torch
import torch.utils.data
import torchvision
from PIL import Image
from diffusers.utils import make_image_grid
from torchvision.io import read_image

from labml import tracker, lab, monit, experiment
from labml.internal.experiment import experiment_singleton, ModelSaver

from labml.configs import BaseConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.train_valid import ModeState, hook_model_outputs
from labml_nn.gan.stylegan import (
    Discriminator,
    Generator,
    GradientPenalty,
    PathLengthPenalty,
)
from labml_nn.gan.wasserstein import DiscriminatorLoss, GeneratorLoss
from labml_nn.utils import cycle_dataloader

from .synth_dataset import SyntheticDataset, InitialDataset, sample_synthetic_dataset
from .mapping_network import MappingNetwork
from .optimizer_saver import PyTorchOptimizerSaver


class Configs(BaseConfigs):
    """
    ## Configurations
    """

    # Device to train the model on.
    # [`DeviceConfigs`](https://docs.labml.ai/api/helpers.html#labml_helpers.device.DeviceConfigs)
    #  picks up an available CUDA device or defaults to CPU.
    device: torch.device

    # [StyleGAN2 Discriminator](index.html#discriminator)
    discriminator: Discriminator
    # [StyleGAN2 Generator](index.html#generator)
    generator: Generator
    # [Mapping network](index.html#mapping_network)
    mapping_network: MappingNetwork

    # [Label embedding](index.html#label_embedding)
    label_embedding: torch.nn.Embedding
    label_embedding_dim: int = 128

    # Discriminator and generator loss functions.
    # We use [Wasserstein loss](../wasserstein/index.html)
    discriminator_loss: DiscriminatorLoss
    generator_loss: GeneratorLoss

    # Optimizers
    generator_optimizer: torch.optim.Adam
    discriminator_optimizer: torch.optim.Adam
    mapping_network_optimizer: torch.optim.Adam
    label_embedding_optimizer: torch.optim.Adam

    # [Gradient Penalty Regularization Loss](index.html#gradient_penalty)
    gradient_penalty: GradientPenalty
    # Gradient penalty coefficient $\gamma$
    gradient_penalty_coefficient: float = 10.0

    # [Path length penalty](index.html#path_length_penalty)
    path_length_penalty: PathLengthPenalty

    # Data loader
    loader: Iterator

    # Batch size
    batch_size: int = 32
    # Dimensionality of $z$ and $w$
    d_latent: int = 256
    # Height/width of the image
    image_size: int = 64
    # Number of layers in the mapping network
    mapping_network_layers: int = 8
    # Generator & Discriminator learning rate
    learning_rate: float = 1e-3
    # Mapping network learning rate ($100 \times$ lower than the others)
    mapping_network_learning_rate: float = 1e-5
    # Number of steps to accumulate gradients on. Use this to increase the effective batch size.
    gradient_accumulate_steps: int = 1
    # $\beta_1$ and $\beta_2$ for Adam optimizer
    adam_betas: Tuple[float, float] = (0.0, 0.99)
    # Probability of mixing styles
    style_mixing_prob: float = 0.9

    # Total number of training steps
    training_steps: int = 3_000

    # Number of blocks in the generator (calculated based on image resolution)
    n_gen_blocks: int

    # ### Lazy regularization
    # Instead of calculating the regularization losses, the paper proposes lazy regularization
    # where the regularization terms are calculated once in a while.
    # This improves the training efficiency a lot.

    # The interval at which to compute gradient penalty
    lazy_gradient_penalty_interval: int = 4
    # Path length penalty calculation interval
    lazy_path_penalty_interval: int = 32
    # Skip calculating path length penalty during the initial phase of training
    lazy_path_penalty_after: int = 5_000

    # How often to log generated images
    log_generated_interval: int = 200
    # How often to save model checkpoints
    save_checkpoint_interval: int = 2_000

    # Training mode state for logging activations
    mode: ModeState
    # Whether to log model layer outputs
    log_layer_outputs: bool = False

    # <a id="dataset_path"></a>
    # We trained this on [CelebA-HQ dataset](https://github.com/tkarras/progressive_growing_of_gans).
    # You can find the download instruction in this
    # [discussion on fast.ai](https://forums.fast.ai/t/download-celeba-hq-dataset/45873/3).
    # Save the images inside `data/stylegan` folder.
    dataset_path: str = "./data/102flowers/"
    num_classes: int = 102

    # Synthetic dataset
    synth_dataset_num_images: int = 8000
    synth_dataset_batch_size: int = 128

    def __init__(self):
        super().__init__()

        self.device = DeviceConfigs()
        self.gradient_penalty = GradientPenalty()

    def init(
        self,
        dataset,
        discriminator=None,
        generator=None,
        mapping_network=None,
        label_embedding=None,
    ):
        """
        ### Initialize
        """

        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        # Continuous [cyclic loader](../../utils.html#cycle_dataloader)
        self.loader = cycle_dataloader(dataloader)

        # $\log_2$ of image resolution
        log_resolution = int(math.log2(self.image_size))

        # Create discriminator and generator
        if discriminator is not None:
            self.discriminator = discriminator
        else:
            self.discriminator = Discriminator(log_resolution).to(self.device)

        if generator is not None:
            self.generator = generator
        else:
            self.generator = Generator(log_resolution, self.d_latent).to(self.device)

        if mapping_network is not None:
            self.mapping_network = mapping_network
        else:
            self.mapping_network = MappingNetwork(
                in_features=self.d_latent + self.label_embedding_dim,
                out_features=self.d_latent,
                n_layers=self.mapping_network_layers,
            ).to(self.device)

        if label_embedding is not None:
            self.label_embedding = label_embedding
        else:
            self.label_embedding = torch.nn.Embedding(
                num_embeddings=self.num_classes, embedding_dim=self.label_embedding_dim
            ).to(self.device)

        # Get number of generator blocks for creating style and noise inputs
        self.n_gen_blocks = self.generator.n_blocks
        # Create mapping network

        # Create path length penalty loss
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.device)

        # Add model hooks to monitor layer outputs
        if self.log_layer_outputs:
            hook_model_outputs(self.mode, self.discriminator, "discriminator")
            hook_model_outputs(self.mode, self.generator, "generator")
            hook_model_outputs(self.mode, self.mapping_network, "mapping_network")

        # Discriminator and generator losses
        self.discriminator_loss = DiscriminatorLoss().to(self.device)
        self.generator_loss = GeneratorLoss().to(self.device)

        # Create optimizers
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=self.adam_betas,
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.learning_rate, betas=self.adam_betas
        )
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=self.mapping_network_learning_rate,
            betas=self.adam_betas,
        )
        self.label_embedding_optimizer = torch.optim.Adam(
            self.label_embedding.parameters(),
            lr=self.learning_rate,
            betas=self.adam_betas,
        )

        # Set tracker configurations
        tracker.set_image("generated", True)

    def get_w(self, batch_size: int, labels: torch.Tensor = None):
        """
        ### Sample $w$

        This samples $z$ randomly and get $w$ from the mapping network.

        We also apply style mixing sometimes where we generate two latent variables
        $z_1$ and $z_2$ and get corresponding $w_1$ and $w_2$.
        Then we randomly sample a cross-over point and apply $w_1$ to
        the generator blocks before the cross-over point and
        $w_2$ to the blocks after.
        """

        if labels is None:
            labels = torch.randint(0, self.num_classes, (batch_size,)).to(self.device)

        labels_embedding = self.label_embedding(labels)

        # Mix styles
        if torch.rand(()).item() < self.style_mixing_prob:
            # Random cross-over point
            cross_over_point = int(torch.rand(()).item() * self.n_gen_blocks)

            # Sample $z_1$ and $z_2$
            z2 = torch.randn(batch_size, self.d_latent).to(self.device)
            z1 = torch.randn(batch_size, self.d_latent).to(self.device)

            # Concatenate labels embedding to z
            z1_l = torch.cat([z1, labels_embedding], dim=1)
            z2_l = torch.cat([z2, labels_embedding], dim=1)

            # Get $w_1$ and $w_2$
            w1 = self.mapping_network(z1_l)
            w2 = self.mapping_network(z2_l)

            # Expand $w_1$ and $w_2$ for the generator blocks and concatenate
            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.n_gen_blocks - cross_over_point, -1, -1)
            return torch.cat((w1, w2), dim=0), labels
        # Without mixing
        else:
            # Sample $z$ and $z$
            z = torch.randn(batch_size, self.d_latent).to(self.device)

            # Concatenate labels embedding to z
            z_l = torch.cat([z, labels_embedding], dim=1)

            # Get $w$ and $w$
            w = self.mapping_network(z_l)
            # Expand $w$ for the generator blocks
            return w[None, :, :].expand(self.n_gen_blocks, -1, -1), labels

    def get_noise(self, batch_size: int):
        """
        ### Generate noise

        This generates noise for each [generator block](index.html#generator_block)
        """
        # List to store noise
        noise = []
        # Noise resolution starts from $4$
        resolution = 4

        # Generate noise for each generator block
        for i in range(self.n_gen_blocks):
            # The first block has only one $3 \times 3$ convolution
            if i == 0:
                n1 = None
            # Generate noise to add after the first convolution layer
            else:
                n1 = torch.randn(
                    batch_size, 1, resolution, resolution, device=self.device
                )
            # Generate noise to add after the second convolution layer
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)

            # Add noise tensors to the list
            noise.append((n1, n2))

            # Next block has $2 \times$ resolution
            resolution *= 2

        # Return noise tensors
        return noise

    def generate_images(self, batch_size: int, labels: torch.Tensor = None):
        """
        ### Generate images

        This generate images using the generator
        """

        # Get $w$
        w, labels = self.get_w(batch_size, labels=labels)
        # Get noise
        noise = self.get_noise(batch_size)

        # Generate images
        images = self.generator(w, noise)

        # Return images and $w$
        return images, w, labels

    def step(self, idx: int):
        """
        ### Training Step
        """

        # Train the discriminator
        with monit.section("Discriminator"):
            # Reset gradients
            self.discriminator_optimizer.zero_grad()

            # Accumulate gradients for `gradient_accumulate_steps`
            for i in range(self.gradient_accumulate_steps):
                # Update `mode`. Set whether to log activation
                with self.mode.update(
                    is_log_activations=(idx + 1) % self.log_generated_interval == 0
                ):
                    # Get real images from the data loader
                    real_images, real_labels = next(self.loader)

                    real_labels = real_labels.to(self.device)
                    real_images = real_images.to(self.device)

                    # Sample images from generator
                    generated_images, *_ = self.generate_images(
                        self.batch_size, real_labels
                    )
                    # Discriminator classification for generated images
                    fake_output = self.discriminator(generated_images.detach())

                    # We need to calculate gradients w.r.t. real images for gradient penalty
                    if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                        real_images.requires_grad_()

                    # Discriminator classification for real images
                    real_output = self.discriminator(real_images)

                    # Get discriminator loss
                    real_loss, fake_loss = self.discriminator_loss(
                        real_output, fake_output
                    )
                    disc_loss = real_loss + fake_loss

                    # Add gradient penalty
                    if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                        # Calculate and log gradient penalty
                        gp = self.gradient_penalty(real_images, real_output)
                        tracker.add("loss.gp", gp)
                        # Multiply by coefficient and add gradient penalty
                        disc_loss = (
                            disc_loss
                            + 0.5
                            * self.gradient_penalty_coefficient
                            * gp
                            * self.lazy_gradient_penalty_interval
                        )

                    # Compute gradients
                    disc_loss.backward()

                    # Log discriminator loss
                    tracker.add("loss.discriminator", disc_loss)

            if (idx + 1) % self.log_generated_interval == 0:
                # Log discriminator model parameters occasionally
                tracker.add("discriminator", self.discriminator)
                tracker.add("discriminator_optimizer", self.discriminator_optimizer)

            # Clip gradients for stabilization
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), max_norm=1.0
            )
            # Take optimizer step
            self.discriminator_optimizer.step()

        # Train the generator
        with monit.section("Generator"):
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.mapping_network_optimizer.zero_grad()

            # Accumulate gradients for `gradient_accumulate_steps`
            for i in range(self.gradient_accumulate_steps):
                # Sample images from generator
                generated_images, w, _ = self.generate_images(self.batch_size)
                # Discriminator classification for generated images
                fake_output = self.discriminator(generated_images)

                # Get generator loss
                gen_loss = self.generator_loss(fake_output)

                # Add path length penalty
                if (
                    idx > self.lazy_path_penalty_after
                    and (idx + 1) % self.lazy_path_penalty_interval == 0
                ):
                    # Calculate path length penalty
                    plp = self.path_length_penalty(w, generated_images)
                    # Ignore if `nan`
                    if not torch.isnan(plp):
                        tracker.add("loss.plp", plp)
                        gen_loss = gen_loss + plp

                # Calculate gradients
                gen_loss.backward()

                # Log generator loss
                tracker.add("loss.generator", gen_loss)

            if (idx + 1) % self.log_generated_interval == 0:
                # Log discriminator model parameters occasionally
                tracker.add("generator", self.generator)
                tracker.add("generator_optimizer", self.generator_optimizer)

                tracker.add("mapping_network", self.mapping_network)
                tracker.add("mapping_network_optimizer", self.mapping_network_optimizer)

                tracker.add("label_embedding", self.label_embedding)
                tracker.add("label_embedding_optimizer", self.label_embedding_optimizer)

            # Clip gradients for stabilization
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(
                self.mapping_network.parameters(), max_norm=1.0
            )
            torch.nn.utils.clip_grad_norm_(
                self.label_embedding.parameters(), max_norm=1.0
            )

            # Take optimizer step
            self.generator_optimizer.step()
            self.mapping_network_optimizer.step()
            self.label_embedding_optimizer.step()

        # Log generated images
        if (idx + 1) % self.log_generated_interval == 0:
            logged_images = torch.cat([generated_images[:6], real_images[:3]], dim=0)
            samples_path = Path(experiment_singleton().run.run_path) / "samples"
            samples_path.mkdir(parents=True, exist_ok=True)

            tracker.add("generated", logged_images)
            torchvision.utils.save_image(
                logged_images, fp=f"{samples_path}/samples_{idx}.png", nrow=3
            )

        # Save model checkpoints
        if (idx + 1) % self.save_checkpoint_interval == 0:
            experiment.save_checkpoint()

        # Flush tracker
        tracker.save()

    def train(self, global_step=0):
        """
        ## Train model
        """

        # Loop for `training_steps`
        idx = global_step
        for _ in monit.loop(self.training_steps - global_step):
            # Take a training step
            self.step(idx)
            #
            if (idx + 1) % self.log_generated_interval == 0:
                tracker.new_line()

            idx += 1


def train(
    configs,
    step_id,
    dataset,
    global_step=0,
    discriminator=None,
    generator=None,
    mapping_network=None,
    label_embedding=None,
    restore_experiment_uuid=None,
):
    """
    ### Train StyleGAN2
    """
    # Create an experiment
    lab_path = Path(
        f"/scratch/shared/beegfs/dzverev/gen_collaps/stylegan_cond/step_{step_id}"
    )
    lab_path.mkdir(exist_ok=True, parents=True)

    lab.configure({"path": str(lab_path)})

    if restore_experiment_uuid is not None:
        experiment.load(restore_experiment_uuid)
        experiment.create(name="stylegan2", uuid=restore_experiment_uuid)
    else:
        experiment.create(name="stylegan2")

    # Set configurations and override some
    experiment.configs(
        configs, {"device.cuda_device": 0, "log_generated_interval": 200}
    )

    configs.init(
        dataset,
        discriminator=discriminator,
        generator=generator,
        mapping_network=mapping_network,
        label_embedding=label_embedding,
    )

    # Set models for saving and loading
    experiment.add_pytorch_models(
        mapping_network=configs.mapping_network,
        generator=configs.generator,
        discriminator=configs.discriminator,
    )
    experiment.add_model_savers(
        {
            name: PyTorchOptimizerSaver(name, optimizer)
            for name, optimizer in dict(
                generator_optimizer=configs.generator_optimizer,
                discriminator_optimizer=configs.discriminator_optimizer,
                mapping_network_optimizer=configs.mapping_network_optimizer,
                label_embedding_optimizer=configs.label_embedding_optimizer,
            ).items()
        }
    )

    # Start the experiment
    with experiment.start():
        # Run the training loop
        configs.train(global_step)

        synthetic_dataset_path = (
            Path(experiment_singleton().run.run_path) / "synth_dataset"
        )
        sample_synthetic_dataset(configs, synthetic_dataset_path)

    experiment.create()
    return synthetic_dataset_path


def restore_from_previous_step(configs, restore_from):
    experiment_path = Path(restore_from)
    experiment_uuid = experiment_path.name

    """
    Checkpoints are stored in following format:
    /{run_path}/checkpoints/{train_step_id}
    
    Examples:
    /stylegan/logs/269df7125b2211efbb79d94a5e421c0d/checkpoints/123999
    /stylegan/logs/269df7125b2211efbb79d94a5e421c0d/checkpoints/43999
    /stylegan/logs/269df7125b2211efbb79d94a5e421c0d/checkpoints/139999
    
    We take the most recent one with sorting by step_id
    """

    last_checkpoint = sorted(
        map(
            lambda path: int(path.split("/")[-1]),
            glob.glob(f"{experiment_path}/checkpoints/*"),
        ),
        reverse=True,
    )
    global_step = last_checkpoint[0]

    checkpoints_path = experiment_path / "checkpoints" / str(global_step)

    # load checkpoints into models
    configs.generator.load_state_dict(torch.load(checkpoints_path / "generator.pth"))
    configs.discriminator.load_state_dict(
        torch.load(checkpoints_path / "discriminator.pth")
    )
    configs.mapping_network.load_state_dict(
        torch.load(checkpoints_path / "mapping_network.pth")
    )
    configs.label_embedding.load_state_dict(
        torch.load(checkpoints_path / "label_embedding.pth")
    )

    configs.generator_optimizer.load_state_dict(
        torch.load(checkpoints_path / "generator_optimizer.pth")
    )
    configs.discriminator_optimizer.load_state_dict(
        torch.load(checkpoints_path / "discriminator_optimizer.pth")
    )
    configs.mapping_network_optimizer.load_state_dict(
        torch.load(checkpoints_path / "mapping_network_optimizer.pth")
    )
    configs.label_embedding_optimizer.load_state_dict(
        torch.load(checkpoints_path / "label_embedding_optimizer.pth")
    )

    # remove pid
    shutil.rmtree(experiment_path / "pids", ignore_errors=True)

    # restore step from experiment name
    m = re.search("step_([0-9]+)", str(experiment_path))
    step_id = int(m[1])

    print(
        f"""Successfully restored from: {restore_from}
            STEP_ID = {step_id}
            global_step= {global_step}
            """
    )

    return step_id, global_step, experiment_uuid


def main(
    restore_from: str = None,
    restore_synthetic_dataset_path: str = None,
    *,
    TOTAL_STEPS=12,
):
    # Create configurations object
    configs = Configs()
    dataset = InitialDataset(image_size=configs.image_size)

    restore_experiment_uuid = None
    STEP_ID = 0

    if restore_from is None:
        synthetic_dataset_path = train(configs, step_id=STEP_ID, dataset=dataset)

        STEP_ID = 1
        global_step = 0
    else:
        assert (
            restore_synthetic_dataset_path is not None
        ), "restore_synthetic_dataset_path can't be None"
        # We need to reinit config obj to make sure network is ready to be reinitialized
        configs.init(dataset)

        # Restore parameters
        STEP_ID, checkpoint_step, restore_experiment_uuid = restore_from_previous_step(
            configs, restore_from
        )

        global_step = checkpoint_step + 1
        synthetic_dataset_path = restore_synthetic_dataset_path

        # If we got restored from the last step checkpoint, move to the next step
        if global_step >= configs.training_steps - 1:
            print("Model was restored from the last training step. Moving further.")

            STEP_ID += 1
            global_step = 0
            restore_experiment_uuid = None

    for _ in range(TOTAL_STEPS - STEP_ID):
        dataset = SyntheticDataset(configs, synthetic_dataset_path)
        old_discriminator = configs.discriminator
        old_generator = configs.generator
        old_mapping_network = configs.mapping_network
        old_label_embedding = configs.label_embedding

        configs = Configs()
        train(
            configs,
            step_id=STEP_ID,
            dataset=dataset,
            global_step=global_step,
            generator=old_generator,
            discriminator=old_discriminator,
            mapping_network=old_mapping_network,
            label_embedding=old_label_embedding,
            restore_experiment_uuid=restore_experiment_uuid,
        )

        global_step = 0
        STEP_ID += 1


def resample(restore_from):
    assert restore_from, "argument restore_from cannot be None or empty"

    # Create configurations object
    configs = Configs()
    dataset = InitialDataset(image_size=configs.image_size)

    configs.init(dataset)
    _, synthetic_dataset_path = restore_from_previous_step(configs, restore_from)
    sample_synthetic_dataset(configs, synthetic_dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stylegan trainer")
    parser.add_argument(
        "--restore_from", type=str, help="experiment path", default=None
    )
    parser.add_argument(
        "--restore_synthetic_dataset",
        type=str,
        help="synthetic dataset path",
        default=None,
    )
    parser.add_argument("--command", type=str, help="command", default="train")
    args = parser.parse_args()

    if args.command == "resample":
        resample(args.restore_from)
    elif args.command == "train":
        main(args.restore_from, args.restore_synthetic_dataset)
    else:
        raise Exception(f"Wrong command {args.command}")
