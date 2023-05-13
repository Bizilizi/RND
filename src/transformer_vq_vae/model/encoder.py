import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from numpy.random import choice
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
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

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(
            np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)
        backward_indexes = torch.as_tensor(
            np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


class MAEEncoder(torch.nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=2,
        emb_dim=192,
        num_layer=12,
        num_head=3,
        mask_ratios=[0.75, 0.4, 0.1],
        mask_ratios_probs=[0.6, 0.2, 0.2],
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

    def forward(self, img, shuffle: bool = True):
        patches = self.patchify(img)
        patches = rearrange(patches, "b c h w -> (h w) b c")
        patches = patches + self.pos_embedding

        backward_indexes = None
        if shuffle:
            ratio = choice(self.mask_ratios, p=self.mask_ratios_probs)
            self.shuffle.ratio = ratio
            patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat(
            [self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0
        )
        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")

        return features, backward_indexes
