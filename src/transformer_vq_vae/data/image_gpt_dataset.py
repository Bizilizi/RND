import torch
from einops import rearrange
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
            dataset,
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
                # extract patches features
                encoder = vq_vae_model.encoder
                patches = encoder.patchify(x)
                patches = rearrange(patches, "b c h w -> (h w) b c")
                patches = patches + encoder.pos_embedding

                patches = torch.cat(
                    [encoder.cls_token.expand(-1, patches.shape[1], -1), patches],
                    dim=0,
                )
                patches = rearrange(patches, "t b c -> b t c")
                features = encoder.layer_norm(encoder.transformer(patches))
                features = rearrange(features, "b t c -> t b c")

                # quantize features
                (
                    _,
                    quantized_features,
                    *_,
                    input_ids,
                ) = vq_vae_model.feature_quantization(features)
                """
                input_ids shape - T x B x top_k
                """

                # shuffle quantized features
                sos_emb_id = rearrange(input_ids[0], "b k -> 1 b k")
                rest_ids = input_ids[1:]
                """
                sos_emb_id shape - 1 x B x top_k
                rest_ids shape   - T x B x top_k
                """

                encoder.shuffle.ratio = self.ratio
                (
                    masked_input_ids,
                    forward_indexes,
                    backward_indexes,
                ) = encoder.shuffle(rest_ids)
                """
                masked_input_ids shape  - (T*(1 - ratio)) x B x top_k 
                forward_indexes shape   - T x B
                backward_indexes shape  - T x B
                """

                # fill masked patches with learned embedding
                mask_token_id = torch.full(
                    (self.top_k,), self.mask_token, device=device
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
                masked_input_ids = take_indexes(masked_input_ids, backward_indexes)

                # Transform to batch
                masked_input_ids = rearrange(masked_input_ids, "t b k-> b (t k)")
                input_ids = rearrange(input_ids, "t b k -> b (t k)")

                # Add igpt/bert sos token
                masked_input_ids = torch.cat(
                    [
                        torch.full(
                            (masked_input_ids.shape[0], 1),
                            self.sos_token,
                            device=device,
                        ),
                        masked_input_ids,
                    ],
                    dim=1,
                )

                input_ids = torch.cat(
                    [
                        torch.full(
                            (input_ids.shape[0], 1),
                            self.sos_token,
                            device=device,
                        ),
                        input_ids,
                    ],
                    dim=1,
                )

            self.targets.append(y.cpu())
            self.masked_input_ids_values.append(masked_input_ids.cpu())
            self.input_ids_values.append(input_ids.cpu())

        self.targets = torch.cat(self.targets)
        self.masked_input_ids_values = torch.cat(self.masked_input_ids_values)
        self.input_ids_values = torch.cat(self.input_ids_values)

    def __len__(self):
        return len(self.targets)


# if __name__ == "__main__":
#     from src.transformer_vq_vae.model.vit_vq_vae import VitVQVae
#     from src.transformer_vq_vae.utils.wrap_empty_indices import (
#         wrap_dataset_with_empty_indices,
#     )
#     from avalanche.benchmarks import SplitCIFAR10
#     from torchvision.transforms import transforms
#
#     num_class_embeddings = 64
#     num_embeddings = 128
#     sos_token = num_class_embeddings + num_embeddings + 1
#     mask_token = num_class_embeddings + num_embeddings
#
#     benchmark = SplitCIFAR10(
#         n_experiences=5,
#         return_task_id=True,
#         shuffle=True,
#         dataset_root="/Users/ewriji/Desktop/work/RND/datasets",
#         train_transform=transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (1.0, 1.0, 1.0)),
#             ]
#         ),
#         eval_transform=transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (1.0, 1.0, 1.0)),
#             ]
#         ),
#     )
#     vq_vae_model = VitVQVae(
#         num_class_embeddings=num_class_embeddings,
#         num_embeddings=num_embeddings,
#         embedding_dim=192,
#         commitment_cost=0.1,
#         decay=0.001,
#         learning_rate=0.1,
#         weight_decay=1,
#         mask_ratio=0.75,
#         mask_token_id=num_class_embeddings + num_embeddings,
#         use_lpips=True,
#         accelerator="cpu",
#         current_samples_loss_weight=1,
#         batch_size=64,
#         num_epochs=1000,
#         cycle_consistency_weight=1,
#         cycle_consistency_sigma=1000,
#         quantize_features=True,
#         quantize_top_k=1,
#     )
#
#     ImageGPTDataset(
#         vq_vae_model,
#         wrap_dataset_with_empty_indices(benchmark.train_stream[0].dataset, 1),
#         sos_token,
#         mask_token,
#         0.1,
#         top_k=1,
#         num_workers=0,
#     )
