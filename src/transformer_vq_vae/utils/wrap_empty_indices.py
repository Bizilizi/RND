import torch
from avalanche.benchmarks.utils.dataset_definitions import ClassificationDataset
from torch.utils.data import Dataset
from avalanche.benchmarks.utils import make_classification_dataset


class WrappedDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    def __getitem__(self, item):
        x, y, *_ = self.dataset[item]
        data = {
            "images": x,
            "indices": torch.empty(16 * 16 + 1),
        }

        return data, y

    def __len__(self):
        return len(self.dataset)


def wrap_dataset_with_empty_indices(dataset: ClassificationDataset):
    """
    Creates classification dataset compatible with Avalanche framework,
    Takes dataset tuple : (x, y, *_)
    and wraps them into : ({"images": x, "indices": None} , y, *_)
    """

    wrapped_dataset = WrappedDataset(dataset)
    return make_classification_dataset(wrapped_dataset, targets=dataset.targets)
