import random

import torch
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.dataset_definitions import ClassificationDataset
from torch.utils.data import Dataset, ConcatDataset, Subset

from src.qmae_latent_extension.utils.wrap_empty_indices import wrap_dataset


def extend_memory(memory, dataset, num_samples):
    indices = random.choices(list(range(len(dataset))), k=num_samples)
    for ind in indices:
        memory.append(dataset[ind])


def bootstrap_past_samples_from_benchmark(
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

    dataset = wrap_dataset(train_dataset, is_past_domain=True)

    return dataset
