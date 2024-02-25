from functools import partial

import torch
from einops import rearrange
from einops.layers.chainer import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch import nn

from src.vq_vmae_joined_igpt.model.encoder import take_indexes
from src.vq_vmae_joined_igpt.model.image_gpt import ImageGPTConfig, ImageGPTModel
from src.vq_vmae_joined_igpt.model.vit import VisionTransformer


class GPTDecoder(nn.Module):
    def __init__(self, embeddings_dim, embeddings_num, n_positions, patch_size):
        super().__init__()

        configuration = ImageGPTConfig(
            **{
                "activation_function": "quick_gelu",
                "attn_pdrop": 0.1,
                "embd_pdrop": 0.1,
                "initializer_range": 0.02,
                "layer_norm_epsilon": 1e-05,
                "model_type": "imagegpt",
                "n_embd": embeddings_dim,
                "n_head": 4,
                "n_layer": 12,
                "n_positions": n_positions + 1,
                "reorder_and_upcast_attn": False,
                "resid_pdrop": 0.1,
                "scale_attn_by_inverse_layer_idx": False,
                "scale_attn_weights": True,
                "tie_word_embeddings": False,
                "use_cache": False,
                "vocab_size": embeddings_num + 1,
            }
        )

        self.image_gpt = ImageGPTModel(configuration)
        self.ema_sos = nn.Embedding(1, embeddings_dim)

        self.lm_head = nn.Linear(embeddings_dim, embeddings_num, bias=False)
        self.pixel_head = nn.Linear(embeddings_dim, patch_size**2 * 3)

    def forward(self, inputs):
        # Add sos token to inputs
        sos_tokens = self.ema_sos.weight[0][None, None].repeat(inputs.shape[0], 1, 1)
        x = torch.cat([sos_tokens, inputs], dim=1)

        x = self.image_gpt(inputs_embeds=x)

        pixels_output = self.pixel_head(x.last_hidden_state[:, 1:])
        pixels_output = torch.tanh(pixels_output.reshape(-1, 3, 32, 32))

        lm_output = self.lm_head(x.last_hidden_state)

        return pixels_output, lm_output


class VITDecoder(nn.Module):
    def __init__(self, embeddings_dim, embeddings_num, n_positions, patch_size):
        super().__init__()

        self.embeddings_dim = embeddings_dim
        self.base_vit = VisionTransformer(
            img_size=[8],
            patch_size=1,
            embed_dim=embeddings_dim,
            depth=2,
            num_heads=4,
            mlp_ratio=2,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            in_chans=embeddings_dim,
        )
        self.linear = nn.Linear(embeddings_dim, 16 * 3)

    def forward(self, inputs, corrupt_data=True):
        x = inputs.reshape(-1, self.embeddings_dim, 8, 8)
        x, _ = self.base_vit(x, corrupt_data=corrupt_data, return_all_patches=True)
        x = self.linear(x[:, 1:])
        x = x.reshape(-1, 3, 32, 32)
        x = torch.tanh(x)

        return x, None


class MAEDecoder(torch.nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=2,
        emb_dim=192,
        num_layer=4,
        num_head=3,
    ) -> None:
        super().__init__()

        self.mask_token = None
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim)
        )

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size**2)
        self.patch2img = Rearrange(
            "(h w) b (c p1 p2) -> b c (h p1) (w p2)",
            p1=patch_size,
            p2=patch_size,
            h=image_size // patch_size,
        )

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat(
            [
                torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                backward_indexes + 1,
            ],
            dim=0,
        )
        features = torch.cat(
            [
                features,
                self.mask_token.expand(
                    backward_indexes.shape[0] - features.shape[0], features.shape[1], -1
                ),
            ],
            dim=0,
        )
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, "t b c -> b t c")
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask
