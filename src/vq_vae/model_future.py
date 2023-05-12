import torch
from avalanche.benchmarks.utils import make_classification_dataset
from torch.utils.data import Dataset

from src.vq_vae.model.vq_vae import VQVae


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
    vq_vae_model: VQVae,
):
    num_emb, emb_dim = vq_vae_model.vq_vae._embedding.weight.shape
    num_rand_samples = max(num_rand_samples, 256)

    images = []
    for _ in num_rand_samples // 256:
        batch = []
        for _ in range(256):
            batch.append(torch.randperm(num_emb)[:64].to(vq_vae_model.device))

        batch = torch.cat(batch)
        emb = vq_vae_model.vq_vae._embedding(batch).reshape(-1, 8, 8)
        rec = vq_vae_model.decoder(emb[None])

        images.append(rec)

    images = torch.cat(images).cpu()
    return images


def model_future_samples(
    vq_vae_model: VQVae,
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
    elif mode == "uniform_prior":
        generated_images = sample_from_uniform_prior(
            num_rand_samples=num_rand_samples,
            vq_vae_model=vq_vae_model,
        )
    else:
        assert False, "wrong future mode"

    targets = [-2] * generated_images.shape[0]
    tensor_dataset = TensorDataset(generated_images, targets)

    return make_classification_dataset(tensor_dataset, targets=targets)
