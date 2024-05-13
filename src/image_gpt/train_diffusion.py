import os
import random

import math
import pathlib
import typing as t
import torch
from einops import rearrange
from PIL import Image
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.utils import make_grid
from tqdm.auto import tqdm, trange
from transformers import ImageGPTConfig

import wandb
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from src.avalanche.strategies import NaivePytorchLightning
from src.image_gpt.configuration.config import TrainConfig
from src.image_gpt.data.bootstrapped_dataset import BootstrappedDataset
from src.image_gpt.data.image_gpt_dataset import ProjectionsDataset
from src.image_gpt.model.absorbing_diffusion import AbsorbingDiffusion
from src.image_gpt.model.image_gpt import ImageGPTForCausalImageModeling
from src.image_gpt.model.transformer import Transformer
from src.image_gpt.model.vit_vq_vae import VitVQVae
from torch.nn.parallel import DistributedDataParallel as DDP


def get_image_embedding(
    vq_vae_model: VitVQVae,
    config: TrainConfig,
    mask_token: int,
) -> torch.nn.Embedding:
    """
    Created Embedding instance that can take image gpt produced indices and
    simply convert them to tokens suitable for decoder.

    Be careful, id of mask_token have to match with index of image_embeddings.weight.data[-1]
    """
    num_embeddings = vq_vae_model.feature_quantization.num_embeddings

    image_embeddings = torch.nn.Embedding(num_embeddings + 1, config.embedding_dim).to(
        vq_vae_model.device
    )

    image_embeddings.weight.data[
        :num_embeddings
    ] = vq_vae_model.feature_quantization._embedding.weight.data.clone()
    image_embeddings.weight.data[
        mask_token
    ] = vq_vae_model.decoder.mask_token.data.clone()

    return image_embeddings


@torch.no_grad()
def bootstrap_past_samples(
    image_gpt: ImageGPTForCausalImageModeling,
    qmae_model: VitVQVae,
    num_images: int,
    classes_seen_in_past,
    config: TrainConfig,
    transform: t.Optional[t.Any] = None,
) -> ClassificationDataset:
    num_images_per_batch = min(128, num_images)

    """
    Token ids scheme:

    { embeddings tokens } with size = num_embeddings 
    { mask token }        with size = 1
    { sos token }         with size = 1
    { class tokens }      with size = num_classes
    """
    mask_token = qmae_model.feature_quantization.num_embeddings
    sos_token = qmae_model.feature_quantization.num_embeddings + 1

    bootstrapped_dataset = BootstrappedDataset(transform=transform)
    image_embeddings = get_image_embedding(qmae_model, config, mask_token).to(
        qmae_model.device
    )

    for _ in range(num_images // num_images_per_batch):
        images, latent_indices, labels = sample_images(
            diffusion=image_gpt,
            qmae_model=qmae_model,
            embedding=image_embeddings,
            sos_token=sos_token,
            temperature=config.temperature,
            num_images=num_images_per_batch,
            classes_to_sample=classes_seen_in_past,
        )

        _, full_features, _ = qmae_model.encoder(images, return_full_features=True)
        features = qmae_model.get_image_embedding(full_features)

        bootstrapped_dataset.add_data(
            images=images.cpu(),
            latent_indices=latent_indices.cpu(),
            features=features.cpu(),
            labels=labels.cpu(),
        )

    dataset = make_classification_dataset(
        bootstrapped_dataset, targets=bootstrapped_dataset.targets
    )

    return dataset


def train_diffusion(
    *,
    strategy: NaivePytorchLightning,
    config: TrainConfig,
    train_dataset: Dataset,
    device: torch.device,
    classes_seen_so_far,
    num_classes: int,
    is_distributed: bool,
    local_rank: int,
):
    qmae_model = strategy.model
    logger = strategy.train_logger

    num_embeddings = qmae_model.feature_quantization.num_embeddings
    """
    Token ids scheme:
    
    { embeddings tokens } with size = num_embeddings 
    { mask token }        with size = 1
    { sos token }         with size = 1
    { class tokens }      with size = num_classes
    """
    mask_token = num_embeddings
    sos_token = num_embeddings + 1

    """
    vocab_size = num_embeddings + mask_token + igpt_sos_token + num_classes
    """
    vocab_size = num_embeddings + 2 + num_classes

    """
    Length of the token sequence:
    
    { patches tokens + encoder sos } with size = 16 * 16 + 1
    { igpt sos token }               with size = 1
    { class token }                  with size = 1
    """
    n_positions = 16 * 16 + 1

    denoise_fn = Transformer(
        vocab_size=vocab_size,
        codebook_size=num_embeddings,
        embedding_dim=config.embedding_dim,
        block_size=n_positions,
        n_layers=24,
        num_heads=8,
    )
    diffusion_model = AbsorbingDiffusion(
        codebook_size=num_embeddings,
        sequence_length=n_positions,
        total_steps=256,
        loss_type=config.diff_loss_type,
        mask_schedule=config.diff_mask_schedule,
        denoise_fn=denoise_fn,
        mask_id=mask_token,
    )

    # init_token_embeddings(vq_vae_model, image_gpt, config, mask_token)
    image_embeddings = get_image_embedding(qmae_model, config, mask_token).to(
        qmae_model.device
    )

    # Transfer models to corresponding device (DDP)
    diffusion_model = diffusion_model.to(device)
    if is_distributed:
        diffusion_model = DDP(
            diffusion_model, device_ids=[local_rank], output_device=local_rank
        )

    qmae_model.to(device)
    image_embeddings.to(device)

    train_dataset = ProjectionsDataset(
        vq_vae_model=qmae_model,
        dataset=train_dataset,
        sos_token=sos_token,
        mask_token=mask_token,
        ratio=config.diff_masking_ratio,
        num_workers=config.num_workers,
    )
    data_loader = DataLoader(
        train_dataset,
        batch_size=config.diff_batch_size,
        shuffle=True,
    )

    epoch_num = config.diff_num_epochs_max
    grad_scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(
        diffusion_model.parameters(), lr=config.diff_learning_rate
    )

    # exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     learning_rate_schedule(
    #         500, epoch_num * len(data_loader) // config.diff_accumulate_grad_batches
    #     ),
    # )

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    step = 0

    for i in trange(0, epoch_num):
        counter = i
        logger.log_metrics({"diffusion_epoch": counter}, step=step)

        for batch in tqdm(data_loader):
            step += 1

            masked_input_ids = batch["input_ids"][:, 2:].to(device)
            with torch.autocast(device_type=config.accelerator):
                loss, vb_loss = diffusion_model.train_iter(masked_input_ids)
                grad_scaler.scale(loss).backward()

            if step % config.diff_accumulate_grad_batches == 0:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # exp_lr_scheduler.step()

            if local_rank == 0:
                logger.log_metrics(
                    {
                        f"train/diffusion_loss/experience_step_{strategy.experience_step}": loss,
                        "epoch": i,
                    },
                    step=i,
                )

                logger.log_metrics(
                    {
                        f"train/diffusion_vb_loss/experience_step_{strategy.experience_step}": vb_loss,
                        "epoch": i,
                    },
                    step=i,
                )

        # Generate sampled images at the end of the epoch
        if local_rank == 0:
            if is_distributed:
                sampler = diffusion_model.module
            else:
                sampler = diffusion_model

            sample = sample_images(
                diffusion=sampler,
                qmae_model=qmae_model,
                embedding=image_embeddings,
                sos_token=sos_token,
                return_grid_only=True,
                temperature=config.temperature,
                classes_to_sample=classes_seen_so_far,
            ).cpu()

            if isinstance(logger, WandbLogger):
                logger.log_metrics(
                    {
                        f"train/dataset/experience_step_{strategy.experience_step}/diffusion_samples": wandb.Image(
                            sample.permute(1, 2, 0).numpy()
                        ),
                        "epoch": i,
                    }
                )
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(
                    f"train/dataset/experience_step_{strategy.experience_step}/diffusion_samples",
                    sample / 255,
                    i,
                )

        # Save diffusion model on every epoch
        if local_rank == 0:
            if is_distributed:
                state_dict = diffusion_model.module.state_dict()
            else:
                state_dict = diffusion_model.state_dict()

            for k, v in state_dict.items():
                state_dict[k] = v.cpu()

            torch.save(
                state_dict,
                f"{config.checkpoint_path}/diffusion-exp{strategy.experience_step}-{i}.ckpt",
            )

    if is_distributed:
        return diffusion_model.module
    else:
        return diffusion_model


@torch.no_grad()
def sample_images(
    diffusion,
    qmae_model,
    embedding,
    sos_token,
    classes_to_sample,
    temperature=1.23,
    num_images=8 * 4 * 10,
    return_grid_only=False,
):
    diffusion.eval()
    qmae_model.eval()

    device = qmae_model.device
    decoder = qmae_model.decoder

    diffusion.to(device)
    decoder.to(device)

    diffusion_output = diffusion.sample(
        n_samples=num_images, temp=temperature, sample_steps=256
    )
    diffusion_output[diffusion_output >= sos_token] = 0

    quantized = rearrange(embedding(diffusion_output), "b t c -> t b c")
    features = quantized + decoder.pos_embedding

    features = rearrange(features, "t b c -> b t c")
    features = decoder.transformer(features)
    features = rearrange(features, "b t c -> t b c")
    features = features[1:]  # remove global feature

    patches = decoder.head(features)
    x_recon = decoder.patch2img(patches)

    if return_grid_only:
        grid_image = make_grid(
            x_recon.cpu().data,
        )
        grid_image = (grid_image * 0.266 + 0.4733) * 255
        grid_image = grid_image.clip(0, 255)

        return grid_image
    else:
        return x_recon, diffusion_output, None
