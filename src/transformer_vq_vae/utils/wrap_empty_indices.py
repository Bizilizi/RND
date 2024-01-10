import torch
from avalanche.benchmarks.utils.dataset_definitions import ClassificationDataset
from torch.utils.data import Dataset
from avalanche.benchmarks.utils import make_classification_dataset

from src.transformer_vq_vae.configuration.config import TrainConfig


class WrappedDataset(Dataset):
    def __init__(self, dataset, num_neighbours: int, num_patches: int):
        super().__init__()

        self.dataset = dataset
        self.num_neighbours = num_neighbours
        self.num_patches = num_patches

    def __getitem__(self, item):
        x, y, *_ = self.dataset[item]
        data = {
            "images": x,
            "indices": torch.zeros(
                (self.num_patches * self.num_patches + 1) * self.num_neighbours,
                dtype=torch.int64,
            ),
        }

        return data, y

    def __len__(self):
        return len(self.dataset)


def wrap_dataset_with_empty_indices(
    dataset: ClassificationDataset, num_neighbours: int, config: TrainConfig
):
    """
    Creates classification dataset compatible with Avalanche framework,
    Takes dataset tuple : (x, y, *_)
    and wraps them into : ({"images": x, "indices": None} , y, *_)
    """

    # Derive num patches based on path algorithm from VIT
    num_patches = (config.image_size // config.patch_size) ** 2

    wrapped_dataset = WrappedDataset(dataset, num_neighbours, num_patches)
    return make_classification_dataset(wrapped_dataset, targets=dataset.targets)
