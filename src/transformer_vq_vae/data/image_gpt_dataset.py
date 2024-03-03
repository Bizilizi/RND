import torch
from einops import rearrange, repeat
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

from src.transformer_vq_vae.model.encoder import take_indexes


class ImageGPTDataset(Dataset):
    def __init__(
        self, vq_vae_model, dataset, sos_token, mask_token, ratio, top_k, num_workers=4
    ):
        super().__init__()

        self.sos_token = sos_token
        self.mask_token = mask_token
        self.ratio = ratio
        self.num_workers = num_workers
        self.top_k = top_k

        self.vq_vae_model = vq_vae_model
        self.dataset = dataset

    def __getitem__(self, item):
        data, y, *_ = self.dataset[item]
        image = data["images"]

        input_ids = self._project_image(image)
        input_ids = self._extend_with_sos_token(input_ids)

        return {
            "input_ids": input_ids[0],
            "labels": -1,
        }

    def __len__(self):
        return len(self.dataset)

    @torch.no_grad()
    def _project_image(
        self,
        x,
    ):
        # extract patches features
        encoder = self.vq_vae_model.encoder
        masked_features, _, backward_indexes = encoder(
            x[None].to(self.vq_vae_model.device),
            return_full_features=False,
            ratio=self.ratio,
        )

        # quantize features
        (
            _,
            quantized_features,
            *_,
            masked_input_ids,
        ) = self.vq_vae_model.feature_quantization(masked_features)

        # fill masked patches with learned embedding
        mask_token_id = torch.full(
            (self.top_k,), self.mask_token, device=self.vq_vae_model.device
        )
        mask_token_id = rearrange(mask_token_id, "k -> 1 1 k")

        backward_indexes = torch.cat(
            [
                torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                backward_indexes + 1,
            ],
            dim=0,
        )
        masked_input_ids = torch.cat(
            [
                masked_input_ids,
                mask_token_id.expand(
                    backward_indexes.shape[0] - masked_input_ids.shape[0],
                    masked_input_ids.shape[1],
                    -1,
                ),
            ],
            dim=0,
        )
        masked_input_ids = take_indexes(masked_input_ids, backward_indexes)

        # Transform to batch
        masked_input_ids = rearrange(masked_input_ids, "t b k-> b (t k)")

        return masked_input_ids

    @torch.no_grad()
    def _extend_with_sos_token(self, input_ids):
        sos_tokens = torch.full(
            (input_ids.shape[0], 1),
            self.sos_token,
            device=input_ids.device,
        )

        input_ids = torch.cat([sos_tokens, input_ids], dim=1)

        return input_ids
