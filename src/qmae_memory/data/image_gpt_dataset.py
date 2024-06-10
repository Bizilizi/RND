import torch
from einops import rearrange
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

from src.qmae_memory.model.encoder import take_indexes


class ImageGPTDataset(Dataset):
    def __init__(self, qmae_model, dataset, sos_token, mask_token, num_workers=4):
        super().__init__()

        self.sos_token = sos_token
        self.mask_token = mask_token
        self.num_workers = num_workers

        self.qmae_model = qmae_model
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data, y, *_ = self.dataset[item]

        image = data["images"].to(self.qmae_model.device)

        input_ids = self._project_image(image)
        input_ids = self._extend_with_class_token(input_ids, y)
        input_ids = self._extend_with_sos_token(input_ids)

        return {
            "input_ids": input_ids,
            "labels": y,
        }

    @torch.no_grad()
    def _project_image(self, image):
        # extract pathes featues
        x = image[None]

        encoder = self.qmae_model.encoder
        full_features, backward_indexes = encoder(x, ratio=0)

        (
            *_,
            input_ids,
            _,
        ) = self.qmae_model.feature_quantization(full_features)
        input_ids = rearrange(input_ids, "(t b) 1 -> t b", b=x.shape[0]).squeeze()

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
    def _extend_with_class_token(
        self,
        input_ids,
        class_token,
    ):
        class_tokens = torch.tensor(
            [self.sos_token + 1 + class_token],
            device=input_ids.device,
        )
        """
        Since classes start from 0, we need to shift them by 1 
        to avoid clashing with igpt sos token.
        """

        input_ids = torch.cat([class_tokens, input_ids], dim=0)

        return input_ids
