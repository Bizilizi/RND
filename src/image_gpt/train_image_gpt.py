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
from src.image_gpt.data.projection_dataset import ProjectionsDataset
from src.image_gpt.model.image_gpt import ImageGPTForCausalImageModeling
from src.image_gpt.model.vit_vq_vae import VitVQVae
from torch.nn.parallel import DistributedDataParallel as DDP


def init_token_embeddings(
    vq_vae_model: VitVQVae,
    image_gpt: ImageGPTForCausalImageModeling,
    config: TrainConfig,
    mask_token: int,
) -> None:
    """
    Initialize image gpt token embeddings with vq_vae embeddings.
    We copy data for the first config.num_embeddings from
    VQ-Vae model, and the rest of two, corresponds to mask_token and sos_token
    """

    image_gpt.transformer.wte.weight.data[
        : vq_vae_model.feature_quantization.num_embeddings
    ] = vq_vae_model.feature_quantization._embedding.weight.data.clone()
    image_gpt.transformer.wte.weight.data[
        mask_token
    ] = vq_vae_model.decoder.mask_token.data.clone()


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
    temperature: float,
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
            image_gpt=image_gpt,
            vq_vae_model=qmae_model,
            embedding=image_embeddings,
            sos_token=sos_token,
            temperature=temperature,
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


def learning_rate_schedule(warmup_steps, total_steps):
    """Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps"""

    def learning_rate_fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return learning_rate_fn


def train_igpt(
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

    """
    Token ids scheme:
    
    { embeddings tokens } with size = num_embeddings 
    { mask token }        with size = 1
    { sos token }         with size = 1
    { class tokens }      with size = num_classes
    """
    mask_token = qmae_model.feature_quantization.num_embeddings
    sos_token = qmae_model.feature_quantization.num_embeddings + 1

    """
    vocab_size = num_embeddings + mask_token + igpt_sos_token + num_classes
    """
    vocab_size = qmae_model.feature_quantization.num_embeddings + 2 + num_classes

    """
    Length of the token sequence:
    
    { patches tokens + encoder sos } with size = 16 * 16 + 1
    { igpt sos token }               with size = 1
    { class token }                  with size = 1
    """
    n_positions = 16 * 16 + 1 + 1 + 1

    configuration = ImageGPTConfig(
        **{
            "activation_function": "quick_gelu",
            "attn_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-05,
            "model_type": "imagegpt",
            "n_embd": config.embedding_dim,
            "n_head": config.igpt_num_heads,
            "n_layer": config.igpt_num_layers,
            "n_positions": n_positions,
            "reorder_and_upcast_attn": False,
            "resid_pdrop": 0.1,
            "scale_attn_by_inverse_layer_idx": False,
            "scale_attn_weights": True,
            "tie_word_embeddings": False,
            "use_cache": False,
            "vocab_size": vocab_size,
        }
    )
    image_gpt = ImageGPTForCausalImageModeling(configuration)

    # init_token_embeddings(vq_vae_model, image_gpt, config, mask_token)
    image_embeddings = get_image_embedding(qmae_model, config, mask_token).to(
        qmae_model.device
    )

    # Transfer models to corresponding device (DDP)
    image_gpt = image_gpt.to(device)
    if is_distributed:
        image_gpt = DDP(image_gpt, device_ids=[local_rank], output_device=local_rank)

    qmae_model.to(device)
    image_embeddings.to(device)

    train_dataset = ProjectionsDataset(
        vq_vae_model=qmae_model,
        dataset=train_dataset,
        sos_token=sos_token,
        mask_token=mask_token,
        ratio=config.igpt_mask_ratio,
        num_workers=config.num_workers,
    )
    data_loader = DataLoader(
        train_dataset,
        batch_size=config.igpt_batch_size,
        shuffle=True,
    )

    epoch_num = config.igpt_num_epochs_max
    grad_scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(image_gpt.parameters(), lr=config.igpt_learning_rate)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    step = 0

    for i in trange(0, epoch_num):
        counter = i
        logger.log_metrics({"igpt_epoch": counter}, step=step)

        for batch in tqdm(data_loader):
            step += 1

            masked_input_ids = batch["input_ids"].to(device)
            with torch.autocast(device_type=config.accelerator):
                output = image_gpt(input_ids=masked_input_ids)
                loss = loss_fn(
                    output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
                    masked_input_ids[..., 1:].reshape(-1),
                )
                grad_scaler.scale(loss).backward()

            if step % config.igpt_accumulate_grad_batches == 0:
                torch.nn.utils.clip_grad_norm_(image_gpt.parameters(), 10)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if local_rank == 0:
                logger.log_metrics(
                    {
                        f"train/image_gpt_loss/experience_step_{strategy.experience_step}": loss,
                        "epoch": i,
                    },
                    step=i,
                )

        # exp_lr_scheduler.step()

        # Generate sampled images at the end of the epoch
        if local_rank == 0:
            if is_distributed:
                sampler = image_gpt.module
            else:
                sampler = image_gpt

            sample = sample_images(
                image_gpt=sampler,
                vq_vae_model=qmae_model,
                embedding=image_embeddings,
                sos_token=sos_token,
                return_grid_only=True,
                temperature=config.temperature,
                classes_to_sample=classes_seen_so_far,
            ).cpu()

            if isinstance(logger, WandbLogger):
                logger.log_metrics(
                    {
                        f"train/dataset/experience_step_{strategy.experience_step}/igpt_samples": wandb.Image(
                            sample.permute(1, 2, 0).numpy()
                        ),
                        "epoch": i,
                    }
                )
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(
                    f"train/dataset/experience_step_{strategy.experience_step}/igpt_samples",
                    sample / 255,
                    i,
                )

        # Save igpt model on every epoch
        if local_rank == 0:
            if is_distributed:
                state_dict = image_gpt.module.state_dict()
            else:
                state_dict = image_gpt.state_dict()

            for k, v in state_dict.items():
                state_dict[k] = v.cpu()

            torch.save(
                state_dict,
                f"{config.checkpoint_path}/igpt-exp{strategy.experience_step}-{i}.ckpt",
            )

    if is_distributed:
        return image_gpt.module
    else:
        return image_gpt


@torch.no_grad()
def sample_images(
    image_gpt,
    vq_vae_model,
    embedding,
    sos_token,
    classes_to_sample,
    temperature=1.23,
    num_images=8 * 4 * 10,
    return_grid_only=False,
):
    image_gpt.eval()
    vq_vae_model.eval()

    device = vq_vae_model.device
    decoder = vq_vae_model.decoder

    labels = torch.tensor(
        random.choices(classes_to_sample, k=num_images), device=device
    )

    labels_tokens = sos_token + 1 + labels
    sos_tokens = torch.full((num_images,), sos_token, device=device)

    context = torch.cat(
        [
            rearrange(sos_tokens, "n -> n 1"),
            rearrange(labels_tokens, "n -> n 1"),
        ],
        dim=1,
    )

    igpt_output = image_gpt.generate(
        input_ids=context,
        max_length=16 * 16 + 3,
        temperature=temperature,
        do_sample=True,
        top_k=45,
        top_p=0.9,
    )
    igpt_output = igpt_output[:, 2:]
    igpt_output[igpt_output >= sos_token] = 0

    quantized = rearrange(embedding(igpt_output), "b t c -> t b c")
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
        return x_recon, igpt_output, labels
