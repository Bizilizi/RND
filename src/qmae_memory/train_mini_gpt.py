import random

from torch.nn import functional as F
import typing as t
import torch
from einops import rearrange
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm.auto import tqdm, trange
from transformers import ImageGPTConfig

import wandb
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from src.avalanche.strategies import NaivePytorchLightning
from src.qmae_memory.configuration.config import TrainConfig
from src.qmae_memory.data.bootstrapped_dataset import BootstrappedDataset
from src.qmae_memory.data.image_gpt_dataset import GPTDataset
from src.qmae_memory.model.transformer.image_gpt import ImageGPTForCausalImageModeling
from src.qmae_memory.model.qmae import QMAE
from torch.nn.parallel import DistributedDataParallel as DDP

from src.qmae_memory.model.transformer.mingpt import GPT


def init_token_embeddings(
    vq_vae_model: QMAE,
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
    vq_vae_model: QMAE,
    config: TrainConfig,
    mask_token: int,
) -> torch.nn.Embedding:
    """
    Created Embedding instance that can take image gpt produced indices and
    simply convert them to tokens suitable for decoder.

    Be careful, id of mask_token have to match with index of image_embeddings.weight.data[-1]
    """
    num_embeddings = vq_vae_model.feature_quantization.num_embeddings

    image_embeddings = torch.nn.Embedding(
        num_embeddings + 1, config.enc_embedding_dim
    ).to(vq_vae_model.device)

    image_embeddings.weight.data[
        :num_embeddings
    ] = vq_vae_model.feature_quantization._embedding.weight.data.clone()
    image_embeddings.weight.data[
        mask_token
    ] = vq_vae_model.decoder.mask_token.data.clone()

    return image_embeddings


@torch.no_grad()
def bootstrap_past_samples(
    gpt_model: GPT,
    qmae_model: QMAE,
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
        latent_indices, labels = sample_images(
            transformer=gpt_model,
            qmae_model=qmae_model,
            embedding=image_embeddings,
            sos_token=sos_token,
            temperature=config.temperature,
            classes_to_sample=classes_seen_in_past,
            accelerator=config.accelerator,
            num_images=num_images_per_batch,
        )

        bootstrapped_dataset.add_data(
            latent_indices=latent_indices.cpu(),
            labels=labels.cpu(),
        )

    dataset = make_classification_dataset(
        bootstrapped_dataset, targets=bootstrapped_dataset.targets
    )

    return dataset


def get_mini_gpt_optimizer(mini_gpt, learning_rate):
    """
    Following minGPT:
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in mini_gpt.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # special case the position embedding parameter in the root GPT module as not decayed
    no_decay.add("pos_emb")

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in mini_gpt.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": 0.01,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
    return optimizer


def train_mini_gpt(
    *,
    strategy: NaivePytorchLightning,
    config: TrainConfig,
    train_dataset: Dataset,
    device: torch.device,
    classes_seen_so_far,
    num_classes: int,
    n_layer: int = 12,
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

    mini_gpt = GPT(
        vocab_size, n_positions, n_layer=n_layer, n_embd=config.enc_embedding_dim
    )

    # init_token_embeddings(vq_vae_model, image_gpt, config, mask_token)
    image_embeddings = get_image_embedding(qmae_model, config, mask_token).to(
        qmae_model.device
    )

    # Transfer models to corresponding device (DDP)
    mini_gpt = mini_gpt.to(device)
    if is_distributed:
        mini_gpt = DDP(mini_gpt, device_ids=[local_rank], output_device=local_rank)

    qmae_model.to(device)
    image_embeddings.to(device)

    train_dataset = GPTDataset(
        qmae_model=qmae_model,
        dataset=train_dataset,
        sos_token=sos_token,
        mask_token=mask_token,
        num_workers=config.num_workers,
        mask_ratio=config.gpt_mask_ratio,
    )
    data_loader = DataLoader(
        train_dataset,
        batch_size=config.gpt_batch_size,
        shuffle=True,
    )

    if strategy.experience_step < 2:
        epoch_num = config.gpt_num_epochs_max
    else:
        epoch_num = config.gpt_num_epochs_min

    grad_scaler = torch.cuda.amp.GradScaler()
    optimizer = get_mini_gpt_optimizer(mini_gpt, learning_rate=config.gpt_learning_rate)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    step = 0
    for i in trange(0, epoch_num):
        counter = i
        logger.log_metrics({"igpt_epoch": counter}, step=step)

        for batch in tqdm(data_loader):
            step += 1

            input_ids = batch["input_ids"].to(device)
            with torch.autocast(device_type=config.accelerator, dtype=torch.float16):
                output = mini_gpt(input_ids)
                loss = loss_fn(
                    output[:, :-1].reshape(-1, output.shape[-1]),
                    input_ids[..., 1:].reshape(-1),
                )

            grad_scaler.scale(loss).backward()

            if step % config.gpt_accumulate_grad_batches == 0:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if local_rank == 0:
                logger.log_metrics(
                    {
                        f"train/gpt_loss/experience_step_{strategy.experience_step}": loss,
                        "epoch": i,
                    },
                    step=i,
                )
        # Generate sampled images at the end of the epoch
        if local_rank == 0:
            if is_distributed:
                sampler = mini_gpt.module
            else:
                sampler = mini_gpt

            sample = sample_images(
                transformer=sampler,
                qmae_model=qmae_model,
                embedding=image_embeddings,
                sos_token=sos_token,
                return_grid_only=True,
                temperature=config.temperature,
                classes_to_sample=classes_seen_so_far,
                accelerator=config.accelerator,
            ).cpu()

            if isinstance(logger, WandbLogger):
                logger.log_metrics(
                    {
                        f"train/dataset/experience_step_{strategy.experience_step}/gpt_samples": wandb.Image(
                            sample.permute(1, 2, 0).numpy()
                        ),
                        "epoch": i,
                    }
                )
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(
                    f"train/dataset/experience_step_{strategy.experience_step}/gpt_samples",
                    sample / 255,
                    i,
                )

        # Save igpt model on every epoch
        if local_rank == 0:
            if is_distributed:
                state_dict = mini_gpt.module.state_dict()
            else:
                state_dict = mini_gpt.state_dict()

            for k, v in state_dict.items():
                state_dict[k] = v.cpu()

            torch.save(
                state_dict,
                f"{config.checkpoint_path}/gpt-exp{strategy.experience_step}-{i}.ckpt",
            )

    if is_distributed:
        return mini_gpt.module
    else:
        return mini_gpt


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float("Inf")
    return out


@torch.no_grad()
def sample_images(
    transformer,
    qmae_model,
    embedding,
    sos_token,
    classes_to_sample,
    accelerator,
    temperature=1.23,
    num_images=8 * 4 * 10,
    return_grid_only=False,
    top_k=20,
    sample=True,
):
    transformer.eval()
    qmae_model.eval()

    device = qmae_model.device
    decoder = qmae_model.decoder

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
    steps = 16 * 16 + 1
    with torch.autocast(device_type=accelerator, dtype=torch.float16):
        for k in range(steps):
            logits = transformer(context)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = top_k_logits(logits, top_k)

            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)

            # append to the sequence and continue
            context = torch.cat((context, ix), dim=1)

    context = context[:, 2:]
    context[context >= sos_token] = 0

    quantized = rearrange(embedding(context), "b t c -> t b c")
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
        return context, labels
