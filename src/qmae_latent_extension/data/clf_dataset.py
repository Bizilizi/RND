import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.qmae_latent_extension.model.vit_vq_vae import VitVQVae


class ClassificationDataset(Dataset):
    def __init__(
        self,
        vq_vae_model: VitVQVae,
        dataset: Dataset,
    ):
        super().__init__()

        self.targets = []
        self.embeddings = []

        self.vq_vae_model = vq_vae_model
        self.dataset = dataset

        self._project_dataset(vq_vae_model, dataset)

    def __getitem__(self, item):
        return {
            "labels": self.targets[item],
            "embeddings": self.embeddings[item],
        }

    def _project_dataset(
        self,
        vq_vae: VitVQVae,
        dataset: Dataset,
    ):
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        for batch in tqdm(dataloader, leave=False):
            x, y, *_ = batch

            x = x.to(vq_vae.device)

            with torch.no_grad():
                _, full_features, _ = vq_vae.encoder(x)
                image_emb = full_features.mean(dim=0)

                self.targets.append(y.cpu())
                self.embeddings.append(image_emb.cpu())

        self.targets = torch.cat(self.targets)
        self.embeddings = torch.cat(self.embeddings)

    def __len__(self):
        return len(self.targets)
