from torch.utils.data import Dataset
import typing as t
import torch


class BootstrappedDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        experience_step: int,
        classes_seen_in_past,
        transform: t.Optional[t.Any],
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.experience_step = experience_step
        self.classes_seen_in_past = torch.tensor(classes_seen_in_past)
        self.transform = transform

        self.images = None
        self.indices = None
        self.time_indices = None
        self.targets = None

    def add_data(self, images, latent_indices, labels):
        if self.images is None:
            self.images = images
            self.indices = latent_indices
            self.targets = labels
            self.time_indices = torch.isin(labels, self.classes_seen_in_past)
        else:
            self.images = torch.cat([self.images, images], dim=0)
            self.indices = torch.cat([self.indices, latent_indices], dim=0)
            self.targets = torch.cat([self.targets, labels], dim=0)
            self.time_indices = torch.cat(
                [
                    self.time_indices,
                    torch.isin(labels, self.classes_seen_in_past),
                ],
                dim=0,
            )

    def __getitem__(self, item):
        image = self.images[item]
        if self.transform is not None:
            image = self.transform(image)

        data = {
            "images": image,
            "indices": self.indices[item],
            "time_index": self.time_indices[item].item(),
        }
        targets = self.targets[item].item()

        return data, targets

    def __len__(self):
        return len(self.images)
