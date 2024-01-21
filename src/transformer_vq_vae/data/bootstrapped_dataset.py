import typing as t
import torch
from torch.utils.data import Dataset


class BootstrappedDataset(Dataset):
    def __init__(
        self, dataset_path: str, experience_step: int, transform: t.Optional[t.Any]
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.experience_step = experience_step
        self.transform = transform

        self.images = None
        self.indices = None
        self.targets = []

    def add_data(self, images, latent_indices, labels):
        if labels is None:
            """when model is unsupervised"""
            labels = [-1] * images.shape[0]

        if self.images is None:
            self.images = images
            self.indices = latent_indices
            self.targets = labels
        else:
            self.images = torch.cat([self.images, images], dim=0)
            self.indices = torch.cat([self.indices, latent_indices], dim=0)
            self.targets.extend(labels)

    def __getitem__(self, item):
        image = self.images[item]

        if self.transform is not None:
            image = self.transform(image)

        indices = self.indices[item]

        data = {
            "images": image,
            "indices": indices,
        }
        targets = {
            "class": self.targets[item],
            "time_tag": -1,
        }

        return data, targets

    def __len__(self):
        return len(self.images)
