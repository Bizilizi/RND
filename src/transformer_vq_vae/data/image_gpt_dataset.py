import torch
from einops import rearrange
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

from src.transformer_vq_vae.model.encoder import take_indexes


class ImageGPTDataset(Dataset):
    def __init__(
        self, vq_vae_model, dataset, sos_token, mask_token, ratio, num_workers=4
    ):
        super().__init__()

        self.sos_token = sos_token
        self.mask_token = mask_token
        self.ratio = ratio
        self.num_workers = num_workers

        self.input_ids_values = []
        self.masked_input_ids_values = []
        self.targets = []

        self.vq_vae_model = vq_vae_model
        self.dataset = dataset

        self._project_dataset(vq_vae_model, dataset)

    def __getitem__(self, item):
        return {
            "masked_input_ids": self.masked_input_ids_values[item],
            "input_ids": self.input_ids_values[item],
            "labels": self.targets[item],
        }

    @torch.no_grad()
    def _project_dataset(self, vq_vae_model, dataset):
        dataloader = DataLoader(
            ConcatDataset([dataset] * 5),
            batch_size=256,
            shuffle=False,
            num_workers=self.num_workers,
        )
        device = vq_vae_model.device

        for batch in tqdm(dataloader, leave=False):
            data, y, *_ = batch

            x = data["images"]
            x = x.to(device)

            with torch.no_grad():
                # extract pathes featues
                encoder = vq_vae_model.encoder
                patches = encoder.patchify(x)
                patches = rearrange(patches, "b c h w -> (h w) b c")
                patches = patches + encoder.pos_embedding

                patches = torch.cat(
                    [encoder.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0
                )
                patches = rearrange(patches, "t b c -> b t c")
                features = encoder.layer_norm(encoder.transformer(patches))
                features = rearrange(features, "b t c -> t b c")

                # quantize features
                _, quantized_features, _, input_ids = vq_vae_model.feature_quantization(
                    features
                )
                input_ids = rearrange(input_ids, "(t b) 1 -> t b", b=x.shape[0])

                # shuffle quantized features
                sos_emb_id = rearrange(input_ids[0], "b -> 1 b 1")
                rest_ids = rearrange(input_ids[1:], "t b -> t b 1")

                encoder.shuffle.ratio = self.ratio
                masked_input_ids, forward_indexes, backward_indexes = encoder.shuffle(
                    rest_ids
                )

                # fill masked pathes with learned embedding
                mask_token_id = torch.tensor(self.mask_token, device=device)
                mask_token_id = mask_token_id[None, None]

                backward_indexes = torch.cat(
                    [
                        torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                        backward_indexes + 1,
                    ],
                    dim=0,
                )
                masked_input_ids = torch.cat(
                    [
                        sos_emb_id,
                        masked_input_ids,
                        mask_token_id.expand(
                            backward_indexes.shape[0] - masked_input_ids.shape[0] - 1,
                            masked_input_ids.shape[1],
                            -1,
                        ),
                    ],
                    dim=0,
                )
                masked_input_ids = take_indexes(
                    masked_input_ids, backward_indexes
                ).squeeze()

                # Add igpt/bert sos token
                masked_input_ids = torch.cat(
                    [
                        torch.full(
                            (1, masked_input_ids.shape[-1]),
                            self.sos_token,
                            device=device,
                        ),
                        masked_input_ids,
                    ],
                    dim=0,
                )

                input_ids = torch.cat(
                    [
                        torch.full(
                            (1, input_ids.shape[-1]),
                            self.sos_token,
                            device=device,
                        ),
                        input_ids,
                    ],
                    dim=0,
                )

                # Transform to batch
                masked_input_ids = rearrange(masked_input_ids, "t b -> b t")
                input_ids = rearrange(input_ids, "t b -> b t")

            self.targets.append(y.cpu())
            self.masked_input_ids_values.append(masked_input_ids.cpu())
            self.input_ids_values.append(input_ids.cpu())

        self.targets = torch.cat(self.targets)
        self.masked_input_ids_values = torch.cat(self.masked_input_ids_values)
        self.input_ids_values = torch.cat(self.input_ids_values)

    def __len__(self):
        return len(self.targets)
