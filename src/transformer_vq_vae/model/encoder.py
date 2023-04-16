from functools import partial

import torch
from torch import nn
from src.transformer_vq_vae.model.vit import (
    VisionTransformer,
)


class VitEncoder(nn.Module):
    def __init__(self, embeddings_dim, patch_size, corruption_rate):
        super().__init__()

        self.base_vit = VisionTransformer(
            img_size=[32],
            patch_size=patch_size,
            num_layers=12,
            embed_dim=embeddings_dim,
            depth=12,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            corruption_rate=corruption_rate,
        )

    def forward(self, x):
        features, masked_indices = self.base_vit.forward(x, return_all_patches=True)

        return features, masked_indices

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.fc.weight.data.clone()
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        self.fc.weight.copy_(w)
