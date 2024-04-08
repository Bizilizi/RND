import random

import torch
from avalanche.benchmarks.utils.dataset_definitions import ClassificationDataset
from einops import rearrange
from torch.utils.data import Dataset, ConcatDataset, Subset

from src.qmae_latent_extension.utils.wrap_empty_indices import wrap_dataset


class DummyBootstrap(Dataset):
    def __init__(self, vq_vae_model, dataset, num_workers=4):
        super().__init__()
        self.num_workers = num_workers

        self.vq_vae_model = vq_vae_model
        self.dataset = dataset
        self.targets = dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data, y, *_ = self.dataset[item]

        image = data["images"].to(self.vq_vae_model.device)

        input_ids = self._project_image(image)

        data = {
            "images": image.cpu(),
            "indices": input_ids.cpu(),
            "is_past_domain": 1,
        }

        return data, y

    @torch.no_grad()
    def _project_image(self, image):
        # extract pathes featues
        x = image[None]
        _, full_features, _ = self.vq_vae_model.encoder(
            x,
            return_full_features=True,
        )

        (
            *_,
            input_ids,
            _,
        ) = self.vq_vae_model.feature_quantization(full_features)
        input_ids = rearrange(input_ids, "(t b) 1 -> t b", b=x.shape[0]).squeeze()

        return input_ids


def extend_memory(memory, dataset, num_samples):
    indices = random.choices(list(range(len(dataset))), k=num_samples)
    for ind in indices:
        memory.append(dataset[ind])


def bootstrap_past_samples_from_benchmark(
    vq_vae_model,
    num_images: int,
    benchmark,
    experience_step,
) -> ClassificationDataset:
    num_images_per_batch = min(128, num_images)

    train_dataset = ConcatDataset(
        [
            experience.dataset
            for experience in benchmark.train_stream[: experience_step + 1]
        ]
    )
    random_indices = torch.randperm(len(train_dataset))[:num_images_per_batch].int()

    train_dataset = Subset(train_dataset, random_indices)
    train_dataset = DummyBootstrap(vq_vae_model=vq_vae_model, dataset=train_dataset)

    return train_dataset
