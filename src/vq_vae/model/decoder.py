import torch
from torch import nn

from src.vq_vae.model.resnet import ResidualStack
from transformers import (
    ImageGPTConfig,
    ImageGPTModel,
)


class Decoder(nn.Module):
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
            nn.ConvTranspose2d(
                in_channels=num_hiddens,
                out_channels=num_hiddens // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=num_hiddens // 2,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, inputs):
        return torch.tanh(self._model(inputs))


class GPTDecoder(nn.Module):
    def __init__(self, embeddings_dim, embeddings_num, n_positions):
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
                "n_positions": n_positions,
                "reorder_and_upcast_attn": False,
                "resid_pdrop": 0.1,
                "scale_attn_by_inverse_layer_idx": False,
                "scale_attn_weights": True,
                "tie_word_embeddings": False,
                "use_cache": False,
                "vocab_size": embeddings_num,
            }
        )

        self.image_gpt = ImageGPTModel(configuration)
        self.linear = nn.Linear(embeddings_dim, 16 * 3)

    def forward(self, inputs):
        x = self.image_gpt(inputs_embeds=inputs)
        x = self.linear(x.last_hidden_state)
        x = x.reshape(-1, 3, 32, 32)

        return x
