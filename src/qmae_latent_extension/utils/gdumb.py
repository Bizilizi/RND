import random

from torch.utils.data import Dataset


def extend_memory(memory, dataset, num_samples):
    indices = random.choices(list(range(len(dataset))), k=num_samples)
    for ind in indices:
        memory.append(dataset[ind])
