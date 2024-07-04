import numpy as np
import torch
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch import nn


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(
        sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1])
    )


class PatchVITDiscriminator(torch.nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=2,
        emb_dim=192,
        num_layer=12,
        num_head=3,
        mae_encoder=None,
    ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros((image_size // patch_size) ** 2, 1, emb_dim)
        )

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.patch_size = patch_size

        if mae_encoder is not None:
            transformer = mae_encoder.transformer
            layer_norm = mae_encoder.layer_norm
        else:
            transformer = torch.nn.Sequential(
                *[Block(emb_dim, num_head) for _ in range(num_layer)]
            )
            layer_norm = torch.nn.LayerNorm(emb_dim)

        self.discriminator = nn.Sequential(
            transformer,
            layer_norm,
            nn.Linear(emb_dim, 1),
        )

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, img, forward_indexes, remain_T):
        if remain_T == 0:
            return torch.tensor([], device=img.device)

        # create image patches
        patches = self.patchify(img)
        patches = rearrange(patches, "b c h w -> (h w) b c")
        full_patches = patches + self.pos_embedding

        """
        Compute mask for patches based on mask image.
        
        We patchify mask with the same patch size as image to
        figure out which path to leave off.  
        """
        masked_patches = take_indexes(full_patches, forward_indexes)
        masked_patches = masked_patches[:remain_T]
        masked_patches = torch.cat(
            [self.cls_token.expand(-1, masked_patches.shape[1], -1), masked_patches],
            dim=0,
        )
        masked_patches = rearrange(masked_patches, "t b c -> b t c")

        logits = self.discriminator(masked_patches)

        return logits
