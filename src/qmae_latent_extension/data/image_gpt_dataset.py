import torch
from einops import rearrange
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

from src.qmae_latent_extension.model.encoder import take_indexes


class ImageGPTDataset(Dataset):
    def __init__(
        self, vq_vae_model, dataset, sos_token, mask_token, ratio, num_workers=4
    ):
        super().__init__()

        self.sos_token = sos_token
        self.mask_token = mask_token
        self.ratio = ratio
        self.num_workers = num_workers

        self.vq_vae_model = vq_vae_model
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data, y, *_ = self.dataset[item]

        image = data["images"].to(self.vq_vae_model.device)
        time_index = data["time_index"]

        input_ids = self._project_image(image)
        input_ids = self._extend_with_time_index(input_ids, time_index)
        input_ids = self._extend_with_sos_token(input_ids)

        return {
            "input_ids": input_ids,
            "labels": y,
        }

    @torch.no_grad()
    def _project_image(self, image):
        # extract pathes featues
        x = image[None]
        encoder = self.vq_vae_model.encoder
        masked_features, full_features, backward_indexes = encoder(
            x,
            return_full_features=True,
            ratio=self.ratio,
        )

        if self.ratio == 1:
            (
                *_,
                input_ids,
            ) = self.vq_vae_model.feature_quantization(full_features)
            input_ids = rearrange(input_ids, "(t b) 1 -> t b", b=x.shape[0]).squeeze()
        else:
            # quantize features
            (
                *_,
                input_ids,
            ) = self.vq_vae_model.feature_quantization(masked_features)
            input_ids = rearrange(input_ids, "(t b) 1 -> t b 1", b=x.shape[0])

            # fill masked pathes with learned embedding
            mask_token_id = torch.full(
                (1, 1),
                self.mask_token,
                device=self.vq_vae_model.device,
            )

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

        return input_ids

    @torch.no_grad()
    def _extend_with_sos_token(self, input_ids):
        sos_tokens = torch.tensor(
            [self.sos_token],
            device=input_ids.device,
        )

        input_ids = torch.cat([sos_tokens, input_ids], dim=0)

        return input_ids

    @torch.no_grad()
    def _extend_with_time_index(
        self,
        input_ids,
        time_index,
    ):
        time_index_tokens = torch.tensor(
            [time_index + 1 + self.sos_token],
            device=input_ids.device,
        )

        input_ids = torch.cat([time_index_tokens, input_ids], dim=0)

        return input_ids
