from functools import partial

import torch
from torch import nn
from src.transformer_vq_vae.model.vit import (
    VisionTransformer,
)


class VitEncoder(nn.Module):
    def __init__(self, embeddings_dim, patch_size, corruption_rate):
        super().__init__()

        self.corruption_rate = 1 - corruption_rate
        self.base_vit = VisionTransformer(
            img_size=[32],
            patch_size=patch_size,
            embed_dim=embeddings_dim,
            depth=6,
            num_heads=4,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            corruption_rate=0.0,
        )

    def forward(self, x):
        B, nc, w, h = x.shape

        patches = self.base_vit.patch_embed(x)  # patch linear embedding
        _, N, D = patches.shape
        pos_encoding = self.base_vit.interpolate_pos_encoding(N, D, w, h)
        clf_pos_encoding = pos_encoding[:, :1]
        patches_pos_encoding = pos_encoding[:, 1:]

        # Flatten everything to select sub set of patches
        patches += patches_pos_encoding
        patches = patches.flatten(0, 1)

        # Select tokens to be processes, mask everything else
        num_tokens = int(self.corruption_rate * patches.shape[0])
        non_masked_indices = torch.randperm(patches.shape[0])[:num_tokens]

        patches = patches[non_masked_indices]
        patches = patches.reshape(B, -1, D)

        # Add the [CLS] token to the embed patch tokens
        cls_tokens = self.base_vit.cls_token.expand(B, -1, -1) + clf_pos_encoding
        patches = torch.cat([cls_tokens, patches], dim=1)

        features, _ = self.base_vit.forward(patches=patches, return_all_patches=True)

        return features, non_masked_indices
