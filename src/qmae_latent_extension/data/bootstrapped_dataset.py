from torch.utils.data import Dataset
import typing as t
import torch


class BootstrappedDataset(Dataset):
    def __init__(
        self,
        transform: t.Optional[t.Any],
    ):
        super().__init__()

        self.transform = transform

        self.images = None
        self.indices = None
        self.targets = None
        self.features = None

        self.mean = 0
        self.std = 0

    def add_data(self, *, images, latent_indices, features, labels):
        if self.images is None:
            self.images = images
            self.indices = latent_indices
            self.targets = labels
            self.features = features
        else:
            self.images = torch.cat([self.images, images], dim=0)
            self.indices = torch.cat([self.indices, latent_indices], dim=0)
            self.targets = torch.cat([self.targets, labels], dim=0)
            self.features = torch.cat([self.features, features], dim=0)

        self.mean = self.images.mean()
        self.std = self.images.std()

    def __getitem__(self, item):
        image = self.images[item]
        image = (image - self.mean) / self.std

        data = {
            "images": image,
            "indices": self.indices[item],
            "features": self.features[item],
            "is_past_domain": 1,
        }
        targets = self.targets[item].item()

        return data, targets

    def __len__(self):
        return len(self.images)
