from functools import partial

import torch
from torch import nn
from src.vq_vae.model.resnet import ResidualStack
from src.vq_vae.model.vit import VisionTransformer, VisionTransformerWithLinear


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        regularization_dropout,
    ):
        super().__init__()

        self._model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_hiddens // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_hiddens // 2,
                out_channels=num_hiddens,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ResidualStack(
                in_channels=num_hiddens,
                num_hiddens=num_hiddens,
                num_residual_layers=num_residual_layers,
                num_residual_hiddens=num_residual_hiddens,
                regularization_dropout=regularization_dropout,
            ),
        )

    def forward(self, inputs):
        return self._model(inputs)


class VitEncoder(nn.Module):
    def __init__(self, image_embedding_dim, vit_embeddings_dim, vit_patch_size):
        super().__init__()

        self.base_vit = VisionTransformer(
            patch_size=vit_patch_size,
            num_layers=12,
            embed_dim=vit_embeddings_dim,
            depth=12,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.fc = nn.Linear(vit_embeddings_dim, image_embedding_dim)

    def forward(self, x):
        features = self.base_vit.forward(x, return_all_patches=True)
        logits = self.fc(features[:, 0])

        return logits, features

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.fc.weight.data.clone()
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        self.fc.weight.copy_(w)
