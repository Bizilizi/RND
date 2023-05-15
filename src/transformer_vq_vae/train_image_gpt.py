import math
import pathlib

import torch

import wandb
from PIL import Image
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from einops import rearrange
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import trange, tqdm
from transformers import ImageGPTConfig, ImageGPTForCausalImageModeling

from src.avalanche.strategies import NaivePytorchLightning
from torchvision.utils import make_grid
from torchvision.io import read_image

from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.data.image_gpt_dataset import ImageGPTDataset
from src.transformer_vq_vae.model.vit_vq_vae import VitVQVae
from src.vq_vae.model.image_gpt_casual import ImageGPTCausal


class BootstrappedDataset(Dataset):
    def __init__(self, dataset_path: str, experience_step: int):
        super().__init__()

        self.dataset_path = dataset_path
        self.experience_step = experience_step
        self.x = []
        self.targets = []

    def add_images(self, images):
        total_img = len(self.x)
        for i, image in enumerate(images):
            pathlib.Path(f"{self.dataset_path}/exp_{self.experience_step}").mkdir(
                parents=True, exist_ok=True
            )
            image_path = (
                f"{self.dataset_path}/exp_{self.experience_step}/{total_img + i}.png"
            )

            image = self._rescale_image(image)
            im = Image.fromarray(image)
            im.save(image_path)

            self.x.append(image_path)
            self.targets.append(-1)

    @staticmethod
    def _rescale_image(image):
        image = (image + 0.5) * 255
        image = torch.clamp(image, 0, 255)
        image = image.permute(1, 2, 0).to("cpu", torch.uint8)
        image = image.numpy()

        return image

    def __getitem__(self, item):
        image = read_image(self.x[item])
        image = image / 255
        return image, self.targets[item]

    def __len__(self):
        return len(self.x)


@torch.no_grad()
def bootstrap_past_samples(
    image_gpt: ImageGPTForCausalImageModeling,
    vq_vae_model: VitVQVae,
    num_images: int,
    experience_step: int,
    dataset_path: str,
    config: TrainConfig,
    temperature: float = 1.0,
) -> ClassificationDataset:
    num_images = num_images * experience_step
    num_images_per_batch = min(128, num_images)

    bootstrapped_dataset = BootstrappedDataset(
        dataset_path=dataset_path, experience_step=experience_step
    )

    decoder = vq_vae_model.decoder
    image_embeddings = torch.nn.Embedding(
        config.num_embeddings + 1, config.embedding_dim
    )
    image_embeddings.weight.data[:-1] = vq_vae_model.vq_vae._embedding.weight.data
    image_embeddings.weight.data[-1] = vq_vae_model.decoder.mask_token.data

    for _ in range(num_images // num_images_per_batch):
        context = torch.full(
            (num_images_per_batch, 1), 1, device=image_gpt.device
        )  # initialize with SOS token
        output = image_gpt.generate(
            input_ids=context,
            max_length=16 * 16 + 2,
            temperature=temperature,
            do_sample=True,
            top_k=45,
        )

        output = output[:, 1:]
        # output[output == (config.num_embeddings + 1)] = 0

        quantized = rearrange(image_embeddings(output), "b t c -> t b c")
        features = quantized + decoder.pos_embedding

        features = rearrange(features, "t b c -> b t c")
        features = decoder.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # remove global feature

        patches = decoder.head(features)
        recon = decoder.patch2img(patches)

        bootstrapped_dataset.add_images(recon.cpu())

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
    n_layer: int = 12,
):
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
            "n_positions": 16 * 16 + 2,
            "reorder_and_upcast_attn": False,
            "resid_pdrop": 0.1,
            "scale_attn_by_inverse_layer_idx": False,
            "scale_attn_weights": True,
            "tie_word_embeddings": False,
            "use_cache": False,
            "vocab_size": config.num_embeddings + 2,
        }
    )
    image_gpt = ImageGPTForCausalImageModeling(configuration)
    image_gpt.transformer.wte.weight.data[
        :-1
    ] = strategy.model.vq_vae._embedding.weight.data

    vq_vae_model = strategy.model
    logger = strategy.train_logger

    old_mask_ratios = vq_vae_model.encoder.mask_ratios
    old_mask_ratios_probs = vq_vae_model.encoder.mask_ratios_probs
    vq_vae_model.encoder.mask_ratios = [0.5, 0.4, 0.35]
    vq_vae_model.encoder.mask_ratios_probs = [0.4, 0.3, 0.3]

    train_dataset = ImageGPTDataset(
        vq_vae_model=vq_vae_model,
        dataset=train_dataset,
        sos_token=config.num_embeddings,
        num_embeddings=config.num_embeddings,
    )
    data_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        [
            {"params": list(image_gpt.parameters())[1:], "lr": 0.001},
            {"params": image_gpt.transformer.wte.parameters(), "lr": 0.001},
        ]
    )
    exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, learning_rate_schedule(500, 3 * len(data_loader) // 2)
    )

    weights = torch.ones(config.num_embeddings + 1)
    weights[-1] = 0.7
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights).to(device)

    image_embeddings = torch.nn.Embedding(
        config.num_embeddings + 1, config.embedding_dim
    )
    image_embeddings.weight.data[:-1] = vq_vae_model.vq_vae._embedding.weight.data
    image_embeddings.weight.data[-1] = vq_vae_model.decoder.mask_token.data

    vq_vae_model.to(device)
    image_gpt.to(device)
    image_embeddings.to(device)

    step = 0
    for i in trange(0, 3):
        counter = i
        logger.log_metrics({"epoch": counter}, step=step)

        for batch in tqdm(data_loader):
            step += 1

            input_ids = batch["input_ids"].to(device)
            output = image_gpt(input_ids=input_ids)
            loss = loss_fn(
                output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
                input_ids[..., 1:].reshape(-1),
            )
            loss.backward()

            if step % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()
                exp_lr_scheduler.step()

            logger.log_metrics(
                {
                    f"train/image_gpt_loss/experience_step_{strategy.experience_step}": loss,
                    "epoch": i,
                },
                step=i,
            )

            if step % 1000 == 0:
                sample = get_sample_image(
                    image_gpt,
                    vq_vae_model.decoder,
                    image_embeddings,
                    device,
                ).cpu()
                sample = sample / 255

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

    vq_vae_model.encoder.mask_ratios = old_mask_ratios
    vq_vae_model.encoder.mask_ratios_probs = old_mask_ratios_probs

    return image_gpt


def get_sample_image(
    image_gpt,
    decoder,
    embedding,
    device,
    num_images=8 * 4 * 10,
    return_output=False,
):
    image_gpt.eval()

    with torch.no_grad():

        context = torch.full(
            (num_images, 1), 1, device=device
        )  # initialize with SOS token
        output = image_gpt.generate(
            input_ids=context,
            max_length=16 * 16 + 2,
            temperature=1,
            do_sample=True,
            top_k=45,
        )

        output = output[:, 1:]
        # output[output == (config.num_embeddings + 1)] = 0

        quantized = rearrange(embedding(output), "b t c -> t b c")
        features = quantized + decoder.pos_embedding

        features = rearrange(features, "t b c -> b t c")
        features = decoder.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # remove global feature

        patches = decoder.head(features)
        x_recon = decoder.patch2img(patches)

    grid_image = make_grid(
        x_recon.cpu().data,
    )
    grid_image = (grid_image + 0.5) * 255
    grid_image = grid_image.clip(0, 255)

    if return_output:
        return grid_image, output
    else:
        return grid_image