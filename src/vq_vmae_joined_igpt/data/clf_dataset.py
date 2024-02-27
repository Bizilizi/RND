import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.vq_vmae_joined_igpt.model.vq_vmae_joined_igpt import VQVMAEJoinedIgpt


class ClassificationDataset(Dataset):
    def __init__(
        self,
        vq_vae_model: VQVMAEJoinedIgpt,
        dataset: Dataset,
        depth: int = 8,
        use_igpt: bool = False,
    ):
        super().__init__()

        self.targets = []
        self.embeddings = []

        self.vq_vae_model = vq_vae_model
        self.dataset = dataset
        self.depth = depth
        self.use_igp = use_igpt

        self._project_dataset(vq_vae_model, dataset)

    def __getitem__(self, item):
        return {
            "labels": self.targets[item],
            "embeddings": self.embeddings[item],
        }

    @torch.no_grad()
    def _project_dataset(
        self,
        vq_vae: VQVMAEJoinedIgpt,
        dataset: Dataset,
    ):
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

        # Cut transformer depth
        old_transformer_seq = vq_vae.encoder.transformer
        vq_vae.encoder.transformer = vq_vae.encoder.transformer[: self.depth]

        for batch in tqdm(dataloader, leave=False):
            x, y, *_ = batch
            x = x.to(vq_vae.device)

            with torch.no_grad():
                _, full_features, _ = vq_vae.encoder(x)
                if self.use_igp:
                    (*_, z_indices, _) = vq_vae.feature_quantization(
                        full_features, return_distances=True
                    )
                    z_indices = rearrange(z_indices, "t b k -> b (t k)")
                    input_ids = vq_vae._rand_mask_indices(z_indices)

                    if vq_vae.supervised and y is not None:
                        input_ids = vq_vae._extend_with_classes(y, input_ids)

                    input_ids = vq_vae._extend_with_sos_token(input_ids)
                    igpt_output = vq_vae.image_gpt(
                        input_ids=input_ids, output_hidden_states=True
                    )
                    image_emb = igpt_output.hidden_states[-1].mean(1)
                else:
                    image_emb = full_features.mean(dim=0)

                self.targets.append(y.cpu())
                self.embeddings.append(image_emb.cpu())

        self.targets = torch.cat(self.targets)
        self.embeddings = torch.cat(self.embeddings)

        # Restore transformer depth
        vq_vae.encoder.transformer = old_transformer_seq

    def __len__(self):
        return len(self.targets)
