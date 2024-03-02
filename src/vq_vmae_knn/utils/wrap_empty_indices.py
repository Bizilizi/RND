import torch
from avalanche.benchmarks.utils.dataset_definitions import ClassificationDataset
from torch.utils.data import Dataset
from avalanche.benchmarks.utils import make_classification_dataset


class WrappedDataset(Dataset):
    def __init__(self, dataset, num_neighbours: int, time_tag):
        super().__init__()

        self.dataset = dataset
        self.num_neighbours = num_neighbours
        self.time_tag = time_tag

    def __getitem__(self, item):
        x, y, *_ = self.dataset[item]
        data = {
            "images": x,
            "indices": torch.zeros(
                (16 * 16 + 1) * self.num_neighbours, dtype=torch.int64
            ),
        }

        targets = {
            "class": y,
            "time_tag": self.time_tag,
        }

        return data, targets

    def __len__(self):
        return len(self.dataset)


def wrap_dataset_with_empty_indices(
    dataset: ClassificationDataset, num_neighbours: int, time_tag: int = 0
):
    """
    Creates classification dataset compatible with Avalanche framework,
    Takes dataset tuple : (x, y, *_)
    and wraps them into : ({"images": x, "indices": None} , y, *_)
    """

    wrapped_dataset = WrappedDataset(dataset, num_neighbours, time_tag)
    return make_classification_dataset(wrapped_dataset, targets=dataset.targets)
