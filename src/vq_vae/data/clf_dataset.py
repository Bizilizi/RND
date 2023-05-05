from torch.utils.data import Dataset, DataLoader
import torch
from transformers import ImageGPTForCausalImageModeling

from src.vq_vae.model.image_gpt_casual import ImageGPTCausal
from src.vq_vae.model.vq_vae import VQVae


class ClassificationDataset(Dataset):
    def __init__(
        self,
        vq_vae_model: VQVae,
        igpt: ImageGPTForCausalImageModeling,
        dataset: Dataset,
        level: int = 6,
    ):
        super().__init__()

        self.targets = []
        self.embeddings = []

        self.vq_vae_model = vq_vae_model
        self.igpt = igpt
        self.dataset = dataset

        self._project_dataset(vq_vae_model, igpt, dataset, level)

    def __getitem__(self, item):
        return {
            "labels": self.targets[item],
            "embeddings": self.embeddings[item],
        }

    def _project_dataset(
        self, vq_vae: VQVae, igpt: ImageGPTCausal, dataset: Dataset, level: int
    ):
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
                output = igpt(input_ids=encoding_indices, output_hidden_states=True)
                image_embeddings = output.hidden_states[level].mean(1)

                self.targets.append(y.cpu())
                self.embeddings.append(image_embeddings.cpu())

        self.targets = torch.cat(self.targets)
        self.embeddings = torch.cat(self.embeddings)

    def __len__(self):
        return len(self.targets)
