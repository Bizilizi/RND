import torch
from avalanche.benchmarks.utils.dataset_definitions import ClassificationDataset
from torch.utils.data import Dataset
from avalanche.benchmarks.utils import make_classification_dataset


class WrappedDataset(Dataset):
    def __init__(self, dataset, time_index):
        super().__init__()

        self.dataset = dataset
        self.time_index = time_index

    def __getitem__(self, item):
        x, y, *_ = self.dataset[item]
        data = {
            "images": x,
            "indices": torch.zeros(16 * 16 + 1, dtype=torch.int64),
            "time_index": self.time_index,
        }

        return data, y

    def __len__(self):
        return len(self.dataset)


def wrap_dataset_with_empty_indices(dataset: ClassificationDataset, time_index):
    """
    Creates classification dataset compatible with Avalanche framework,
    Takes dataset tuple : (x, y, *_)
    and wraps them into : ({"images": x, "indices": None} , y, *_)
    """

    wrapped_dataset = WrappedDataset(dataset, time_index)
    return make_classification_dataset(wrapped_dataset, targets=dataset.targets)
