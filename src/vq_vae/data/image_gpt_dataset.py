from torch.utils.data import Dataset, DataLoader
import torch

from src.vq_vae.model.vq_vae import VQVae


class ImageGPTDataset(Dataset):
    def __init__(self, vq_vae_model: VQVae, dataset, sos_token, with_latents=False):
        super().__init__()

        self.sos_token = sos_token
        self.with_latents = with_latents
        self.values = []
        self.targets = []
        self.embeddigns = []

        self.vq_vae_model = vq_vae_model
        self.dataset = dataset

        self._project_dataset(vq_vae_model, dataset)

    def __getitem__(self, item):
        if not self.with_latents:
            return {"input_ids": self.values[item], "labels": self.targets[item]}
        else:
            return {
                "input_ids": self.values[item],
                "labels": self.targets[item],
                "latents": self.embeddigns[item],
            }

    def _project_dataset(self, vq_vae: VQVae, dataset: Dataset):
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        for batch in dataloader:
            x, y, *_ = batch
            x = x.to(vq_vae.device)

            with torch.no_grad():
                z = vq_vae.encoder(x)
                z = vq_vae.pre_vq_conv(z)
                *_, quantized, _, encoding_indices = vq_vae.vq_vae(z)
                encoding_indices = encoding_indices.squeeze().reshape(
                    x.shape[0], z.shape[2] * z.shape[3]
                )

                self.values.append(encoding_indices.cpu())
                self.targets.append(y.cpu())

                if self.with_latents:
                    self.embeddigns.append(quantized.cpu())

        self.targets = torch.cat(self.targets)

        self.values = torch.cat(self.values)
        self.values = torch.cat(
            [torch.full((self.values.shape[0], 1), self.sos_token), self.values], dim=-1
        )
        self.values = self.values.cpu()

        if self.with_latents:
            self.embeddigns = torch.cat(self.embeddigns)

    def __len__(self):
        return len(self.values)
