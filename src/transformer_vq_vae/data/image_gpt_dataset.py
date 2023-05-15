from einops import rearrange
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch

from src.transformer_vq_vae.model.encoder import take_indexes


class ImageGPTDataset(Dataset):
    def __init__(self, vq_vae_model, dataset, sos_token, num_embeddings):
        super().__init__()

        self.values = []
        self.targets = []
        self.embeddigns = []

        self.vq_vae_model = vq_vae_model
        self.dataset = dataset

        self._project_dataset(vq_vae_model, dataset, num_embeddings, sos_token)

    def __getitem__(self, item):
        return {"input_ids": self.values[item], "labels": self.targets[item]}

    def _project_dataset(self, vq_vae_model, dataset, num_embeddings, sos_token):
        dataloader = DataLoader(
            ConcatDataset([dataset] * 5), batch_size=256, shuffle=False, num_workers=0
        )
        for batch in dataloader:
            x, y, *_ = batch
            x = x.to(vq_vae_model.device)

            with torch.no_grad():
                # extract pathes featues
                features, backward_indexes = vq_vae_model.encoder(x)
                _, quantized_features, _, input_ids = vq_vae_model.vq_vae(features)
                input_ids = rearrange(input_ids, "(b t) 1 -> t b 1", b=x.shape[0])

                # fill masked pathes with learned embedding
                mask_token_id = torch.tensor(
                    num_embeddings, device=vq_vae_model.device
                )[None, None]

                backward_indexes = torch.cat(
                    [
                        torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                        backward_indexes + 1,
                    ],
                    dim=0,
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        mask_token_id.expand(
                            backward_indexes.shape[0] - input_ids.shape[0],
                            input_ids.shape[1],
                            -1,
                        ),
                    ],
                    dim=0,
                )
                input_ids = take_indexes(input_ids, backward_indexes).squeeze()
                input_ids = torch.cat(
                    [
                        torch.full(
                            (1, input_ids.shape[-1]),
                            sos_token,
                            device=vq_vae_model.device,
                        ),
                        input_ids,
                    ],
                    dim=0,
                )
                input_ids = rearrange(input_ids, "t b -> b t")

            self.values.append(input_ids.cpu())
            self.targets.append(y.cpu())

        self.targets = torch.cat(self.targets)

        self.values = torch.cat(self.values)
        self.values = self.values.cpu()

    def __len__(self):
        return len(self.values)
