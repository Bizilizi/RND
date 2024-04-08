import random

import torch
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.dataset_definitions import ClassificationDataset
from einops import rearrange
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
from tqdm.auto import tqdm

from src.qmae_latent_extension.data import bootstrapped_dataset
from src.qmae_latent_extension.utils.wrap_empty_indices import wrap_dataset


class DummyBootstrap(Dataset):
    def __init__(self, vq_vae_model, dataset, num_workers=4, batch_size=256):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.vq_vae_model = vq_vae_model
        self.dataset = dataset

        self.targets = []
        self.images = []
        self.indices = []

        self._project_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = {
            "images": self.images[item],
            "indices": self.indices[item],
            "is_past_domain": 1,
        }

        return data, self.targets[item].item()

    @torch.no_grad()
    def _project_dataset(self):
        dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        for images, y, *_ in tqdm(dataloader, leave=False):
            indices = self._project_batch(images.to(self.vq_vae_model.device))

            self.targets.append(y)
            self.images.append(images)
            self.indices.append(indices)

        self.targets = torch.cat(self.targets).cpu()
        self.images = torch.cat(self.images).cpu()
        self.indices = torch.cat(self.indices).cpu()

    @torch.no_grad()
    def _project_batch(self, batch):
        # extract pathes featues
        x = batch

        _, full_features, _ = self.vq_vae_model.encoder(
            x,
            return_full_features=True,
        )

        (
            *_,
            input_ids,
            _,
        ) = self.vq_vae_model.feature_quantization(full_features)

        input_ids = rearrange(input_ids, "(t b) 1 -> t b", b=x.shape[0])
        input_ids = rearrange(input_ids, "t b -> b t")

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
    train_dataset = ConcatDataset(
        [
            experience.dataset
            for experience in benchmark.train_stream[: experience_step + 1]
        ]
    )
    targets = torch.cat(
        [
            torch.tensor(experience.dataset.targets)
            for experience in benchmark.train_stream[: experience_step + 1]
        ]
    )
    random_indices = torch.randperm(len(train_dataset))[:num_images].int()

    train_dataset = Subset(train_dataset, random_indices)
    train_dataset = DummyBootstrap(vq_vae_model=vq_vae_model, dataset=train_dataset)
    train_dataset = make_classification_dataset(
        train_dataset, targets=targets[random_indices]
    )

    return train_dataset
