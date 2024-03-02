import torch
from einops import rearrange, repeat
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

from src.transformer_vq_vae.model.encoder import take_indexes


class ImageGPTDataset(Dataset):
    def __init__(
        self,
        vq_vae_model,
        dataset,
        sos_token,
        mask_token,
        ratio,
        top_k,
        num_tokens_without_sos,
        num_workers=4,
    ):
        super().__init__()

        self.sos_token = sos_token
        self.mask_token = mask_token
        self.ratio = ratio
        self.num_workers = num_workers
        self.top_k = top_k
        self.num_tokens_without_sos = num_tokens_without_sos

        self.input_ids_values = []
        self.targets = []

        self.vq_vae_model = vq_vae_model
        self.dataset = dataset

        self._construct_dataset(vq_vae_model, dataset)

    @torch.no_grad()
    def _construct_dataset(self, vq_vae_model, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            num_workers=self.num_workers,
        )
        device = vq_vae_model.device

        for batch in tqdm(dataloader, leave=False, desc="Igpt project dataset"):
            data, targets, *_ = batch

            time_tag = targets["time_tag"].to(device)
            indices = data["indices"].to(device)
            y = targets["class"].to(device)
            x = data["images"].to(device)

            current = time_tag == 0
            past = time_tag == -1

            # Add indices related to current CL step
            if current.any():
                current_input_ids = self._project_images(x[current])
                if self.vq_vae_model.supervised:
                    current_input_ids = self._extend_with_classes(
                        y[current],
                        current_input_ids,
                        device,
                    )
                current_input_ids = self._extend_with_sos_token(
                    current_input_ids,
                    device,
                )

                self.targets.append(y[current].cpu())
                self.input_ids_values.append(current_input_ids.cpu())

            # Add indices related to previous CL steps
            if past.any():
                past_input_ids = indices[past]
                if self.vq_vae_model.supervised:
                    past_input_ids = self._extend_with_classes(
                        y[past],
                        past_input_ids,
                        device,
                    )
                past_input_ids = self._extend_with_sos_token(
                    past_input_ids,
                    device,
                )

                self.targets.append(y[past].cpu())
                self.input_ids_values.append(past_input_ids.cpu())

        self.targets = torch.cat(self.targets)
        self.input_ids_values = torch.cat(self.input_ids_values)

    @torch.no_grad()
    def _project_images(self, x):
        forward_output = self.vq_vae_model.forward(x)
        return forward_output.z_indices

    @torch.no_grad()
    def _extend_with_classes(self, y, input_ids, device):
        classes_ids = y + self.num_tokens_without_sos
        """
        Shift classes ids with (num_embeddings + class_token + mask_token)
        to get classes ids.
        """

        classes_ids = repeat(
            classes_ids,
            "b -> b 1",
        ).to(device)

        input_ids = torch.cat([classes_ids, input_ids], dim=1)

        return input_ids

    @torch.no_grad()
    def _extend_with_sos_token(self, input_ids, device):
        sos_tokens = torch.full(
            (input_ids.shape[0], 1),
            self.sos_token,
            device=device,
        )

        input_ids = torch.cat([sos_tokens, input_ids], dim=1)

        return input_ids

    def _rand_mask_indices(self, indices):
        T = indices.shape[0]
        change_T = int(T * self.ratio)

        forward_indices = torch.randperm(T)
        masked_indices = indices.clone()
        masked_indices[forward_indices[:change_T]] = self.mask_token

        return masked_indices

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        input_ids = self.input_ids_values[item]
        masked_input_ids = self._rand_mask_indices(input_ids)

        return {
            "input_ids": input_ids,
            "masked_input_ids": masked_input_ids,
            "labels": self.targets[item],
        }
