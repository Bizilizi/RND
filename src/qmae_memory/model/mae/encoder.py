import numpy as np
import timm
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from numpy.random import choice
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def random_indexes_batch(size: int, batch_size: int, device: torch.device):
    forward_indexes = torch.tile(torch.arange(size, device=device), (batch_size, 1))

    # Shuffle each row independently
    for i in range(batch_size):
        forward_indexes[i] = forward_indexes[i][torch.randperm(size, device=device)]

    # Compute backward indexes
    _, backward_indexes = torch.sort(forward_indexes, dim=1)

    forward_indexes = rearrange(forward_indexes, "b t -> t b")
    backward_indexes = rearrange(backward_indexes, "b t -> t b")

    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(
        sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1])
    )


class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        forward_indexes, backward_indexes = random_indexes_batch(
            T, B, device=patches.device
        )

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes, remain_T


class MAEEncoder(torch.nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=2,
        emb_dim=192,
        num_layer=12,
        num_head=3,
        mask_ratios=[0.75, 0.4, 0.1, 0],
        mask_ratios_probs=[0.6, 0.2, 0.1, 0.1],
    ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros((image_size // patch_size) ** 2, 1, emb_dim)
        )
        self.mask_ratios = mask_ratios
        self.mask_ratios_probs = mask_ratios_probs
        self.shuffle = PatchShuffle(self.mask_ratios[0])

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, img, ratio: float = None):
        patches = self.patchify(img)
        patches = rearrange(patches, "b c h w -> (h w) b c")
        full_patches = patches + self.pos_embedding

        # randomly choose ratio of path masking
        if ratio is None:
            self.shuffle.ratio = choice(self.mask_ratios, p=self.mask_ratios_probs)
        else:
            self.shuffle.ratio = ratio

        masked_patches, forward_indexes, backward_indexes, remain_T = self.shuffle(
            full_patches
        )
        masked_patches = torch.cat(
            [self.cls_token.expand(-1, masked_patches.shape[1], -1), masked_patches],
            dim=0,
        )
        masked_patches = rearrange(masked_patches, "t b c -> b t c")
        masked_features = self.layer_norm(self.transformer(masked_patches))
        masked_features = rearrange(masked_features, "b t c -> t b c")

        return masked_features, forward_indexes, backward_indexes, remain_T
