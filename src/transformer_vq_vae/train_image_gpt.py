import os

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
from transformers import ImageGPTConfig, ImageGPTForCausalImageModeling

import wandb
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from src.avalanche.strategies import NaivePytorchLightning
from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.data.image_gpt_dataset import ImageGPTDataset
from src.transformer_vq_vae.model.vit_vq_vae import VitVQVae


class BootstrappedDataset(Dataset):
    def __init__(
        self, dataset_path: str, experience_step: int, transform: t.Optional[t.Any]
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.experience_step = experience_step
        self.transform = transform

        self.images = []
        self.indices = []
        self.targets = []

    def add_data(self, images, latent_indices):
        total_img = len(self.images)
        for i, (image, indices) in enumerate(zip(images, latent_indices)):
            experience_folder = pathlib.Path(
                f"{self.dataset_path}/exp_{self.experience_step}"
            )
            experience_folder.mkdir(parents=True, exist_ok=True)

            tmp = os.environ.get("TMPDIR", "/tmp")
            run_id, dataset_prefix = self.dataset_path.split("/")[-2:]
            tmp_experience_folder = (
                pathlib.Path(tmp)
                / run_id
                / dataset_prefix
                / f"exp_{self.experience_step}"
            )
            tmp_experience_folder.mkdir(parents=True, exist_ok=True)

            # Prepare image path
            image_path = (
                f"{self.dataset_path}/exp_{self.experience_step}/{total_img + i}.png"
            )
            tmp_image_path = f"{tmp_experience_folder}/{total_img + i}.png"

            image = self._rescale_image(image)
            im = Image.fromarray(image)

            # Save image both to tmp and target dir
            im.save(image_path)
            im.save(tmp_image_path)

            # Save indices
            indices_path = (
                f"{self.dataset_path}/exp_{self.experience_step}/{total_img + i}.pt"
            )
            tmp_indices_path = f"{tmp_experience_folder}/{total_img + i}.pt"

            torch.save(indices, indices_path)
            torch.save(indices, tmp_indices_path)

            # Store paths for later access
            self.images.append(tmp_image_path)
            self.indices.append(tmp_indices_path)
            self.targets.append(-1)

    @staticmethod
    def _rescale_image(image):
        image = (image + 0.5) * 255
        image = torch.clamp(image, 0, 255)
        image = image.permute(1, 2, 0).to("cpu", torch.uint8)
        image = image.numpy()

        return image

    def __getitem__(self, item):
        image = read_image(self.images[item])
        image = image / 255
        if self.transform is not None:
            image = self.transform(image)

        indices = torch.load(self.indices[item]).long()

        data = {
            "images": image,
            "indices": indices,
        }
        targets = self.targets[item]

        return data, targets

    def __len__(self):
        return len(self.images)


def init_token_embeddings(
    vq_vae_model: VitVQVae,
    image_gpt: ImageGPTForCausalImageModeling,
    config: TrainConfig,
    mask_token: int,
) -> None:
    """
    Initialize image gpt token embeddings with vq_vae embeddings.
    We copy data for the first config.num_class_embeddings + config.num_embeddings from
    VQ-Vae model, and the rest of two, corresponds to mask_token and sos_token
    """
    image_gpt.transformer.wte.weight.data[
        : config.num_class_embeddings
    ] = (
        vq_vae_model.feature_quantization.class_quantization._embedding.weight.data.clone()
    )

    image_gpt.transformer.wte.weight.data[
        config.num_class_embeddings : -2
    ] = (
        vq_vae_model.feature_quantization.feature_quantization._embedding.weight.data.clone()
    )
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

    image_embeddings = torch.nn.Embedding(
        config.num_class_embeddings + config.num_embeddings + 1, config.embedding_dim
    ).to(vq_vae_model.device)

    image_embeddings.weight.data[
        : config.num_class_embeddings
    ] = (
        vq_vae_model.feature_quantization.class_quantization._embedding.weight.data.clone()
    )
    image_embeddings.weight.data[
        config.num_class_embeddings : -1
    ] = (
        vq_vae_model.feature_quantization.feature_quantization._embedding.weight.data.clone()
    )
    image_embeddings.weight.data[
        mask_token
    ] = vq_vae_model.decoder.mask_token.data.clone()

    return image_embeddings


@torch.no_grad()
def bootstrap_past_samples(
    image_gpt: ImageGPTForCausalImageModeling,
    vq_vae_model: VitVQVae,
    num_images: int,
    experience_step: int,
    dataset_path: str,
    config: TrainConfig,
    sos_token: int,
    mask_token: int,
    transform: t.Optional[t.Any] = None,
) -> ClassificationDataset:
    num_images_per_batch = min(128, num_images)

    bootstrapped_dataset = BootstrappedDataset(
        dataset_path=dataset_path, experience_step=experience_step, transform=transform
    )
    image_embeddings = get_image_embedding(vq_vae_model, config, mask_token).to(
        vq_vae_model.device
    )

    for _ in range(num_images // num_images_per_batch):
        images, latent_indices = sample_images(
            image_gpt=image_gpt,
            vq_vae_model=vq_vae_model,
            embedding=image_embeddings,
            sos_token=sos_token,
            temperature=config.temperature,
            max_length=(16 * 16 + 1) * config.quantize_top_k + 1,
            num_neighbours=config.quantize_top_k,
        )

        bootstrapped_dataset.add_data(
            images=images.cpu(),
            latent_indices=latent_indices.cpu(),
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
    strategy: NaivePytorchLightning,
    config: TrainConfig,
    train_dataset: Dataset,
    device: torch.device,
    sos_token: int,
    mask_token: int,
    n_layer: int = 12,
    image_gpt: ImageGPTForCausalImageModeling = None,
):
    vq_vae_model = strategy.model
    logger = strategy.train_logger
    vocab_size = config.num_class_embeddings + config.num_embeddings + 2

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
            "n_positions": (16 * 16 + 1) * config.quantize_top_k + 1,
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

    init_token_embeddings(vq_vae_model, image_gpt, config, mask_token)
    image_embeddings = get_image_embedding(vq_vae_model, config, mask_token).to(
        vq_vae_model.device
    )

    vq_vae_model.to(device)
    image_gpt.to(device)
    image_embeddings.to(device)

    train_dataset = ImageGPTDataset(
        vq_vae_model=vq_vae_model,
        dataset=train_dataset,
        sos_token=sos_token,
        mask_token=mask_token,
        ratio=config.igpt_mask_ratio,
        num_workers=config.num_workers,
        top_k=config.quantize_top_k,
    )
    data_loader = DataLoader(
        train_dataset,
        batch_size=config.igpt_batch_size,
        shuffle=True,
    )

    if strategy.experience_step < 2:
        epoch_num = 10
    elif strategy.experience_step < 3:
        epoch_num = 7
    else:
        epoch_num = 5

    grad_scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(image_gpt.parameters(), lr=3e-3)
    exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_schedule(
            500, epoch_num * len(data_loader) // config.igpt_accumulate_grad_batches
        ),
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    step = 0
    for i in trange(0, epoch_num):
        counter = i
        logger.log_metrics({"igpt_epoch": counter}, step=step)

        for batch in tqdm(data_loader):
            step += 1

            input_ids = batch["input_ids"].to(device)
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

            logger.log_metrics(
                {
                    f"train/image_gpt_loss/experience_step_{strategy.experience_step}": loss,
                    "epoch": i,
                },
                step=i,
            )

            if step % 1000 == 0:
                with torch.autocast(device_type=config.accelerator):
                    sample = sample_images(
                        image_gpt=image_gpt,
                        vq_vae_model=vq_vae_model,
                        embedding=image_embeddings,
                        sos_token=sos_token,
                        return_grid_only=True,
                        temperature=config.temperature,
                        max_length=(16 * 16 + 1) * config.quantize_top_k + 1,
                        num_neighbours=config.quantize_top_k,
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

            if step % 100 == 0:
                model_ckpt_path = f"{config.checkpoint_path}/igpt-exp{strategy.experience_step}-{i}.ckpt"
                state_dict = image_gpt.state_dict()
                for k, v in state_dict.items():
                    state_dict[k] = v.cpu()

                torch.save(state_dict, model_ckpt_path)

    return image_gpt


@torch.no_grad()
def sample_images(
    image_gpt,
    vq_vae_model,
    embedding,
    sos_token,
    max_length,
    num_neighbours,
    temperature=1.23,
    num_images=8 * 4 * 10,
    return_grid_only=False,
):
    image_gpt.eval()
    vq_vae_model.eval()

    device = vq_vae_model.device
    decoder = vq_vae_model.decoder

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
    igpt_output[igpt_output == sos_token] = 0

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

    if return_grid_only:
        grid_image = make_grid(
            x_recon.cpu().data,
        )
        grid_image = (grid_image + 0.5) * 255
        grid_image = grid_image.clip(0, 255)

        return grid_image
    else:
        return x_recon, igpt_output
