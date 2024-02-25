import os

import math
import pathlib
import random
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
from src.vq_vmae_joined_igpt.model.image_gpt import ImageGPTForCausalImageModeling

import wandb
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from src.avalanche.strategies import NaivePytorchLightning
from src.vq_vmae_joined_igpt.configuration.config import TrainConfig
from src.vq_vmae_joined_igpt.data.bootstrapped_dataset import BootstrappedDataset
from src.vq_vmae_joined_igpt.data.image_gpt_dataset import ImageGPTDataset
from src.vq_vmae_joined_igpt.model.vq_vmae_joined_igpt import VQVMAEJoinedIgpt
from torch.nn.parallel import DistributedDataParallel as DDP


def init_token_embeddings(
    vq_vae_model: VQVMAEJoinedIgpt,
    image_gpt: ImageGPTForCausalImageModeling,
    mask_token: int,
) -> None:
    """
    Initialize image gpt token embeddings with vq_vae embeddings.
    We copy data for the first config.num_class_embeddings + config.num_embeddings from
    VQ-Vae model, and the rest of two, corresponds to mask_token and sos_token
    """
    num_class_embeddings = (
        vq_vae_model.feature_quantization.class_quantization._embedding.weight.shape[0]
    )
    num_features_embeddings = (
        vq_vae_model.feature_quantization.feature_quantization._embedding.weight.shape[
            0
        ]
    )

    image_gpt.transformer.wte.weight.data[
        :num_class_embeddings
    ] = (
        vq_vae_model.feature_quantization.class_quantization._embedding.weight.data.clone()
    )

    image_gpt.transformer.wte.weight.data[
        num_class_embeddings : num_class_embeddings + num_features_embeddings
    ] = (
        vq_vae_model.feature_quantization.feature_quantization._embedding.weight.data.clone()
    )
    image_gpt.transformer.wte.weight.data[
        mask_token
    ] = vq_vae_model.decoder.mask_token.data.clone()


def get_image_embedding(
    vq_vae_model: VQVMAEJoinedIgpt,
    num_embeddings: int,
    embedding_dim: int,
    mask_token: int,
) -> torch.nn.Embedding:
    """
    Created Embedding instance that can take image gpt produced indices and
    simply convert them to tokens suitable for decoder.

    Be careful, id of mask_token have to match with index of image_embeddings.weight.data[-1]
    """
    image_embeddings = torch.nn.Embedding(num_embeddings, embedding_dim).to(
        vq_vae_model.device
    )

    image_embeddings.weight.data[
        : num_embeddings - 1
    ] = vq_vae_model.feature_quantization._embedding.weight[
        : num_embeddings - 1
    ].data.clone()

    image_embeddings.weight.data[
        mask_token
    ] = vq_vae_model.decoder.mask_token.data.clone()

    return image_embeddings


@torch.no_grad()
def bootstrap_past_samples(
    image_gpt: ImageGPTForCausalImageModeling,
    vq_vae_model: VQVMAEJoinedIgpt,
    num_images: int,
    experience_step: int,
    dataset_path: str,
    config: TrainConfig,
    classes_seen_so_far: t.List[int],
    num_classes: int,
    transform: t.Optional[t.Any] = None,
) -> ClassificationDataset:
    num_images_per_batch = min(128, num_images)

    # Derive igpt params based on supervision
    num_all_embeddings = config.num_class_embeddings + config.num_embeddings
    if vq_vae_model.supervised:
        sos_token = num_all_embeddings + num_classes + 1
        mask_token = num_all_embeddings
    else:
        sos_token = num_all_embeddings + 1
        mask_token = num_all_embeddings

    bootstrapped_dataset = BootstrappedDataset(
        dataset_path=dataset_path, experience_step=experience_step, transform=transform
    )
    image_embeddings = get_image_embedding(
        vq_vae_model=vq_vae_model,
        num_embeddings=num_all_embeddings + 1,
        embedding_dim=config.embedding_dim,
        mask_token=mask_token,
    ).to(vq_vae_model.device)

    for _ in trange(num_images // num_images_per_batch):
        images, latent_indices, labels = sample_images(
            image_gpt=image_gpt,
            vq_vae_model=vq_vae_model,
            embedding=image_embeddings,
            sos_token=sos_token,
            mask_token=mask_token,
            temperature=config.temperature,
            max_length=(16 * 16 + 1) * config.quantize_top_k + 1,
            num_neighbours=config.quantize_top_k,
            num_images=num_images_per_batch,
            classes_seen_so_far=classes_seen_so_far,
            num_tokens_without_sos=num_all_embeddings + 1,
        )

        if labels is not None:
            labels = labels.cpu()

        bootstrapped_dataset.add_data(
            images=images.cpu(), latent_indices=latent_indices.cpu(), labels=labels
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
    classes_seen_so_far: t.List[int],
    num_classes: int,
    n_layer: int = 12,
    image_gpt: ImageGPTForCausalImageModeling = None,
    is_distributed: bool,
    local_rank: int,
):
    vq_vae_model = strategy.model
    logger = strategy.train_logger

    # Derive igpt params based on supervision
    num_all_embeddings = config.num_class_embeddings + config.num_embeddings
    if vq_vae_model.supervised:
        sos_token = num_all_embeddings + 1 + num_classes
        mask_token = num_all_embeddings

        vocab_size = num_all_embeddings + 1 + num_classes + 1
        n_positions = (16 * 16 + 1) * config.quantize_top_k + 2
    else:
        sos_token = num_all_embeddings + 1
        mask_token = num_all_embeddings

        vocab_size = num_all_embeddings + 2
        n_positions = (16 * 16 + 1) * config.quantize_top_k + 1

    # Create IGPT instance
    configuration = ImageGPTConfig(
        **{
            "activation_function": "quick_gelu",
            "attn_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-05,
            "model_type": "imagegpt",
            "n_embd": config.embedding_dim,
            "n_head": 8,
            "n_layer": n_layer,
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
    image_gpt = image_gpt or ImageGPTForCausalImageModeling(configuration)
    init_token_embeddings(vq_vae_model, image_gpt, mask_token)

    # Create image embedding token for easier image generation
    image_embeddings = get_image_embedding(
        vq_vae_model=vq_vae_model,
        num_embeddings=num_all_embeddings + 1,
        embedding_dim=config.embedding_dim,
        mask_token=mask_token,
    )

    # Transfer models to corresponding device (DDP)
    image_gpt = image_gpt.to(device)
    if is_distributed:
        image_gpt = DDP(image_gpt, device_ids=[local_rank], output_device=local_rank)

    vq_vae_model.to(device)
    image_embeddings.to(device)

    # Create IGPT dataset
    train_dataset = ImageGPTDataset(
        vq_vae_model=vq_vae_model,
        dataset=train_dataset,
        sos_token=sos_token,
        mask_token=mask_token,
        ratio=config.igpt_mask_ratio,
        num_workers=config.num_workers,
        top_k=config.quantize_top_k,
        num_tokens_without_sos=num_all_embeddings + 1,
    )
    data_loader = DataLoader(
        train_dataset,
        batch_size=config.igpt_batch_size,
        shuffle=True,
    )

    # Init training variables
    epoch_num = config.max_epochs_igpt
    grad_scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(image_gpt.parameters(), lr=3e-3)
    exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_schedule(
            500, epoch_num * len(data_loader) // config.igpt_accumulate_grad_batches
        ),
    )
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Train loop
    step = 0
    for i in trange(0, epoch_num, desc="Igpt-epochs"):
        counter = i
        logger.log_metrics({"igpt_epoch": counter}, step=step)

        for batch in tqdm(data_loader, desc="Igpt-batch", leave=False):
            step += 1

            input_ids = batch["masked_input_ids"].to(device)

            with torch.autocast(device_type=config.accelerator):
                output = image_gpt(input_ids=input_ids)
                loss = loss_fn(
                    output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
                    input_ids[..., 1:].reshape(-1),
                )
                grad_scaler.scale(loss).backward()

            if step % config.igpt_accumulate_grad_batches == 0:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                exp_lr_scheduler.step()

            if isinstance(logger, WandbLogger) and local_rank == 0:
                logger.log_metrics(
                    {
                        f"train/image_gpt_loss/experience_step_{strategy.experience_step}": loss,
                        "epoch": i,
                    },
                    step=i,
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

        # Generate sampled images at the end of the epoch
        if isinstance(logger, WandbLogger) and local_rank == 0:
            if is_distributed:
                sampler = image_gpt.module
            else:
                sampler = image_gpt

            images, *_ = sample_images(
                image_gpt=sampler,
                vq_vae_model=vq_vae_model,
                embedding=image_embeddings,
                sos_token=sos_token,
                mask_token=mask_token,
                temperature=config.temperature,
                max_length=(16 * 16 + 1) * config.quantize_top_k + 1,
                num_neighbours=config.quantize_top_k,
                classes_seen_so_far=classes_seen_so_far,
                num_tokens_without_sos=num_all_embeddings + 1,
            )

            sample = make_grid(images.cpu().data)
            sample = (sample + 0.5) * 255
            sample = sample.clip(0, 255)

            # Log sampled images
            logger.log_metrics(
                {
                    f"train/dataset/experience_step_{strategy.experience_step}/igpt_samples": wandb.Image(
                        sample.permute(1, 2, 0).numpy()
                    ),
                    "epoch": i,
                }
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
    mask_token,
    max_length,
    num_neighbours,
    classes_seen_so_far,
    num_tokens_without_sos,
    temperature=1.23,
    num_images=8 * 4 * 10,
):
    image_gpt.eval()
    vq_vae_model.eval()

    device = vq_vae_model.device
    decoder = vq_vae_model.decoder
    labels = None

    if vq_vae_model.supervised:
        labels = torch.tensor(
            random.choices(classes_seen_so_far, k=num_images), device=device
        )

        labels_tokens = num_tokens_without_sos + labels
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
            max_length=max_length + 1,
            temperature=temperature,
            do_sample=True,
            top_k=45,
            top_p=0.9,
        )

        igpt_output = igpt_output[:, 2:]
    else:

        context = torch.full(
            (num_images, 1), sos_token, device=device
        )  # initialize with SOS token

        igpt_output = image_gpt.generate(
            input_ids=context,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_k=45,
            top_p=0.9,
        )

        igpt_output = igpt_output[:, 1:]

    igpt_output[igpt_output > mask_token] = 0

    quantized = rearrange(
        embedding(igpt_output), "b (t k) c -> t k b c", k=num_neighbours
    )
    quantized = quantized.mean(dim=1)
    features = quantized + decoder.pos_embedding

    features = rearrange(features, "t b c -> b t c")
    features = decoder.transformer(features)
    features = rearrange(features, "b t c -> t b c")
    features = features[1:]  # remove global feature

    patches = decoder.head(features)
    x_recon = decoder.patch2img(patches)

    return x_recon, igpt_output, labels
