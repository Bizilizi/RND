import torch
from torch.utils.data import Dataset

from avalanche.benchmarks.utils import make_classification_dataset
from src.transformer_vq_vae.model.vit_vq_vae import VitVQVae

from einops import rearrange


class TensorDataset(Dataset):
    def __init__(self, x, targets):
        super().__init__()

        self.x = x
        self.targets = targets

    def __getitem__(self, item):
        return self.x[item], self.targets[item]

    def __len__(self):
        return len(self.x)


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
    num_emb, emb_dim = vq_vae_model.vq_vae._embedding.weight.shape
    num_images = max(num_images, 256)
    decoder = vq_vae_model.decoder

    images = []
    for _ in range(num_images // 256):
        batch = []
        for _ in range(256):
            indices = torch.multinomial(
                torch.arange(num_emb, device=vq_vae_model.device).float(),
                16 * 16 + 1,
                replacement=True,
            )
            batch.append(indices[None])

        batch = torch.cat(batch)
        emb = vq_vae_model.vq_vae._embedding(batch)
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
    vq_vae_model: VitVQVae, ratio: float = 0.2, num_images: int = 16
):
    num_emb, emb_dim = vq_vae_model.vq_vae._embedding.weight.shape
    num_images = max(num_images, 256)

    decoder = vq_vae_model.decoder

    images = []
    for _ in range(num_images // 256):
        batch = []
        for _ in range(256):
            num_rand_samples = int((16 * 16 + 1) * ratio)
            emb_indices = torch.multinomial(
                torch.arange(
                    num_emb,
                    device=vq_vae_model.device,
                ).float(),
                num_rand_samples,
                replacement=False,
            )

            emb_positions = torch.multinomial(
                torch.arange(
                    16 * 16 + 1,
                    device=vq_vae_model.device,
                ).float(),
                num_rand_samples,
                replacement=False,
            )

            mask_emb = vq_vae_model.decoder.mask_token[0].repeat(16 * 16 + 1, 1)
            mask_emb[emb_positions] = vq_vae_model.vq_vae._embedding(emb_indices)

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
    tensor_dataset = TensorDataset(generated_images, targets)

    return make_classification_dataset(tensor_dataset, targets=targets)
