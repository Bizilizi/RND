import random

import torch
from torch.utils.data import Dataset


def extend_memory(memory, dataset, num_samples):
    targets = torch.tensor(dataset.targets)
    classes = targets.unique()

    for class_id in classes:
        indices = torch.arange(len(dataset))
        indices = indices[targets == class_id]

        indices = random.choices(indices.tolist(), k=num_samples // len(classes))
        for ind in indices:
            memory.append(dataset[ind])
