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
