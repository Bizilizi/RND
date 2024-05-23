import torch
from avalanche.benchmarks.utils.dataset_definitions import ClassificationDataset
from torch.utils.data import Dataset
from avalanche.benchmarks.utils import make_classification_dataset


class WrappedDataset(Dataset):
    def __init__(self, dataset, is_past_domain: bool, img_embedding_dim: int):
        super().__init__()

        self.dataset = dataset
        self.is_past_domain = int(is_past_domain)
        self.img_embedding_dim = img_embedding_dim

    def __getitem__(self, item):
        x, y, *_ = self.dataset[item]
        data = {
            "images": x,
            "indices": torch.zeros(16 * 16 + 1, dtype=torch.int64),
            "features": torch.zeros(self.img_embedding_dim, dtype=torch.float64),
            "is_past_domain": self.is_past_domain,
        }

        return data, y

    def __len__(self):
        return len(self.dataset)


def wrap_dataset(
    dataset: ClassificationDataset, img_embedding_dim: int, is_past_domain: bool = False
):
    """
    Creates classification dataset compatible with Avalanche framework,
    Takes dataset tuple : (x, y, *_)
    and wraps them into : ({"images": x, "indices": None, "is_past_domain": 1/0 } , y, *_)
    """

    wrapped_dataset = WrappedDataset(dataset, is_past_domain, img_embedding_dim)
    return make_classification_dataset(wrapped_dataset, targets=dataset.targets)
