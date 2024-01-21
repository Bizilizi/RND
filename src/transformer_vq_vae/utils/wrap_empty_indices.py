import torch
from avalanche.benchmarks.utils.dataset_definitions import ClassificationDataset
from torch.utils.data import Dataset
from avalanche.benchmarks.utils import make_classification_dataset

from src.transformer_vq_vae.configuration.config import TrainConfig


class WrappedDataset(Dataset):
    def __init__(self, dataset, num_neighbours: int, num_patches: int, time_tag: int):
        super().__init__()

        self._dataset = dataset
        self._num_neighbours = num_neighbours
        self._num_patches = num_patches
        self._time_tag = time_tag

    def __getitem__(self, item):
        x, y, *_ = self._dataset[item]
        data = {
            "images": x,
            "indices": torch.zeros(
                (self._num_patches + 1) * self._num_neighbours,
                dtype=torch.int64,
            ),
        }

        y = {
            "class": y,
            "time_tag": self._time_tag,
        }

        return data, y

    def __len__(self):
        return len(self._dataset)


def convert_avalanche_dataset_to_vq_mae_dataset(
    dataset: ClassificationDataset,
    num_neighbours: int,
    config: TrainConfig,
    time_tag: int,
):
    """
    Creates classification dataset compatible with Avalanche framework,
    Takes dataset tuple : (x, y, *_)
    and wraps them into : ({"images": x, "indices": [0,0,...]]} , y, *_)
    """

    # Derive num patches based on path algorithm from VIT
    num_patches = (config.image_size // config.patch_size) ** 2

    wrapped_dataset = WrappedDataset(dataset, num_neighbours, num_patches, time_tag)
    return make_classification_dataset(wrapped_dataset, targets=dataset.targets)
