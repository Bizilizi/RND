from functools import partial

import torch
from torch import nn

from transformers import (
    ImageGPTConfig,
    ImageGPTModel,
)

from src.transformer_vq_vae.model.vit import VisionTransformer


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

        self.base_vit = VisionTransformer(
            img_size=[8],
            patch_size=1,
            num_layers=12,
            embed_dim=embeddings_dim,
            depth=12,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.linear = nn.Linear(embeddings_dim, 16 * 3)

    def forward(self, inputs):
        x = self.base_vit(inputs_embeds=inputs, return_all_patches=True)
        x = self.linear(x[:, 1:])
        x = x.reshape(-1, 3, 32, 32)

        return x, None
