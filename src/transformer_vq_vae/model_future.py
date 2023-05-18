import torch
from einops import rearrange
from torch.utils.data import Dataset

from avalanche.benchmarks.utils import make_classification_dataset
from src.transformer_vq_vae.model.vit_vq_vae import VitVQVae


class TensorDataset(Dataset):
    def __init__(self, x, targets):
        super().__init__()

        self.x = x
        self.targets = targets

    def __getitem__(self, item):
        return self.x[item], self.targets[item]

    def __len__(self):
        return len(self.x)


def sample_random_noise(
    num_rand_samples,
    mu: float = 0.0,
    sigma: float = 1.0,
):
    noise = torch.randn(num_rand_samples, 3, 32, 32)

    noise = mu + noise * sigma
    return noise


def sample_from_uniform_prior(
    num_rand_samples,
    vq_vae_model: VitVQVae,
):
    num_emb, emb_dim = vq_vae_model.vq_vae._embedding.weight.shape
    num_rand_samples = max(num_rand_samples, 256)

    images = []
    for _ in range(num_rand_samples // 256):
        batch = []
        for _ in range(256):
            indices = torch.randperm(num_emb)[:64].to(vq_vae_model.device)
            batch.append(indices[None])

        batch = torch.cat(batch)
        emb = vq_vae_model.vq_vae._embedding(batch)
        emb = rearrange(emb, "b (t1 t2) c -> b c t1 t2", t1=8)
        rec = vq_vae_model.decoder(emb)

        images.append(rec)

    images = torch.cat(images).detach().cpu()
    return images


@torch.no_grad()
def model_future_samples(
    vq_vae_model: VitVQVae,
    num_rand_samples: int = 5_000,
    mode: str = "noise",
):
    if mode == "noise":
        generated_images = sample_random_noise(num_rand_samples=num_rand_samples)
    elif mode == "noise_out":
        generated_images = sample_random_noise(
            num_rand_samples=num_rand_samples,
            mu=2.0,
        )
    # elif mode == "uniform_prior":
    #     generated_images = sample_from_uniform_prior(
    #         num_rand_samples=num_rand_samples,
    #         vq_vae_model=vq_vae_model,
    #     )
    else:
        assert False, "wrong future mode"

    targets = [-2] * generated_images.shape[0]
    tensor_dataset = TensorDataset(generated_images, targets)

    return make_classification_dataset(tensor_dataset, targets=targets)
