import torch
from torch.utils.data import Dataset

from avalanche.benchmarks.utils import make_classification_dataset

from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.model.vit_vq_vae import VitVQVae

from einops import rearrange

from src.transformer_vq_vae.utils.wrap_empty_indices import (
    wrap_dataset_with_empty_indices,
)


class TensorDataset(Dataset):
    def __init__(self, x, targets):
        super().__init__()

        self.x = x
        self.targets = targets

    def __getitem__(self, item):
        return self.x[item], self.targets[item]

    def __len__(self):
        return len(self.x)


def get_latent_embedding(
    vq_vae_model: VitVQVae,
    config: TrainConfig,
) -> torch.nn.Embedding:
    """
    Created Embedding instance that can take indices and
    simply convert them to tokens suitable for decoder.
    """

    image_embeddings = torch.nn.Embedding(
        config.num_class_embeddings + config.num_embeddings, config.embedding_dim
    ).to(vq_vae_model.device)

    image_embeddings.weight.data[
        : config.num_class_embeddings
    ] = (
        vq_vae_model.feature_quantization.class_quantization.embedding.weight.data.clone()
    )
    image_embeddings.weight.data[
        config.num_class_embeddings :
    ] = (
        vq_vae_model.feature_quantization.feature_quantization.embedding.weight.data.clone()
    )

    return image_embeddings


@torch.no_grad()
def sample_random_noise(
    num_images,
    mu: float = 0.0,
    sigma: float = 1.0,
):
    noise = torch.randn(num_images, 3, 32, 32)

    noise = mu + noise * sigma
    return noise


@torch.no_grad()
def sample_from_uniform_prior(
    num_images,
    vq_vae_model: VitVQVae,
):
    feature_embedding = (
        vq_vae_model.feature_quantization.feature_quantization.embedding
    )
    class_embedding = vq_vae_model.feature_quantization.class_quantization.embedding

    num_images = max(num_images, 256)
    decoder = vq_vae_model.decoder

    images = []
    for _ in range(num_images // 256):
        feature_ind_batch = []
        class_ind_batch = []

        for _ in range(256):
            feature_indices = torch.multinomial(
                torch.arange(
                    feature_embedding.weight.shape[0], device=vq_vae_model.device
                ).float(),
                16 * 16,
                replacement=True,
            )
            class_indices = torch.multinomial(
                torch.arange(
                    class_embedding.weight.shape[0], device=vq_vae_model.device
                ).float(),
                1,
                replacement=True,
            )

            feature_ind_batch.append(feature_indices[None])
            class_ind_batch.append(class_indices[None])

        feature_ind_batch = torch.cat(feature_ind_batch)
        class_ind_batch = torch.cat(class_ind_batch)

        class_emb = class_embedding(class_ind_batch)
        feature_emb = feature_embedding(feature_ind_batch)

        emb = torch.cat((class_emb, feature_emb), dim=1)
        quantized = rearrange(emb, "b t c -> t b c")
        features = quantized + decoder.pos_embedding

        features = rearrange(features, "t b c -> b t c")
        features = decoder.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # remove global feature

        patches = decoder.head(features)
        rec = decoder.patch2img(patches)

        images.append(rec)

    images = torch.cat(images).detach().cpu()
    return images


@torch.no_grad()
def sample_image_from_sparse_vector(
    vq_vae_model: VitVQVae,
    ratio: float = 0.2,
    num_images: int = 16,
):
    feature_embedding = (
        vq_vae_model.feature_quantization.feature_quantization.embedding
    )
    class_embedding = vq_vae_model.feature_quantization.class_quantization.embedding

    num_images = max(num_images, 256)

    decoder = vq_vae_model.decoder

    images = []
    for _ in range(num_images // 256):
        batch = []

        for _ in range(256):
            num_rand_samples = int(16 * 16 * ratio)
            feature_indices = torch.multinomial(
                torch.arange(
                    feature_embedding.weight.shape[0], device=vq_vae_model.device
                ).float(),
                num_rand_samples,
                replacement=True,
            )
            class_indices = torch.multinomial(
                torch.arange(
                    class_embedding.weight.shape[0], device=vq_vae_model.device
                ).float(),
                1,
                replacement=True,
            )

            emb_positions = torch.multinomial(
                torch.arange(
                    16 * 16,
                    device=vq_vae_model.device,
                ).float(),
                num_rand_samples,
                replacement=False,
            )
            emb_positions += 1

            mask_emb = vq_vae_model.decoder.mask_token[0].repeat(16 * 16 + 1, 1)
            mask_emb[0] = class_embedding(class_indices)
            mask_emb[emb_positions] = feature_embedding(feature_indices)

            batch.append(mask_emb[None])

        batch = torch.cat(batch)

        quantized = rearrange(batch, "b t c -> t b c")
        features = quantized + decoder.pos_embedding

        features = rearrange(features, "t b c -> b t c")
        features = decoder.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # remove global feature

        patches = decoder.head(features)
        x_recon = decoder.patch2img(patches)

        images.append(x_recon)

    images = torch.cat(images).detach().cpu()
    return images


@torch.no_grad()
def model_future_samples(
    vq_vae_model: VitVQVae,
    config: TrainConfig,
    num_images: int = 5_000,
    mode: str = "noise",
):
    if mode == "noise":
        generated_images = sample_random_noise(num_images=num_images)
    elif mode == "noise_out":
        generated_images = sample_random_noise(
            num_images=num_images,
            mu=2.0,
        )
    elif mode == "uniform_prior":
        generated_images = sample_from_uniform_prior(
            num_images=num_images,
            vq_vae_model=vq_vae_model,
        )
    elif mode == "sparse_prior":
        generated_images = sample_image_from_sparse_vector(
            vq_vae_model=vq_vae_model, num_images=num_images, ratio=0.1
        )
    else:
        assert False, "wrong future mode"

    targets = [-2] * generated_images.shape[0]
    tensor_dataset = wrap_dataset_with_empty_indices(
        TensorDataset(generated_images, targets), num_neighbours=config.quantize_top_k
    )

    return tensor_dataset
