import typing as t

import torch
from torch import nn
from torch.nn import functional as F

from src.avalanche.model.cl_model import CLModel
from src.vq_vae.model.decoder import Decoder
from src.vq_vae.model.encoder import Encoder
from src.vq_vae.model.quiantizer import VectorQuantizer, VectorQuantizerEMA


class VQVae(CLModel):
    def __init__(
        self,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay=0,
        data_variance=1,
        learning_rate: float = 1e-3,
        regularization_dropout: float = 0.0,
        regularization_lambda: float = 0.0,
    ):
        super().__init__()

        self._data_variance = data_variance
        self._learning_rate = learning_rate
        self._weight_decay = regularization_lambda
        self._regularization_dropout = regularization_dropout

        self._encoder = Encoder(
            in_channels=3,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            regularization_dropout=regularization_dropout,
        )
        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1
        )
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, decay
            )
        else:
            self._vq_vae = VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )
        self._decoder = Decoder(
            in_channels=embedding_dim,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            regularization_dropout=regularization_dropout,
        )

    def criterion(
        self,
        x: t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        y: t.Optional[torch.Tensor] = None,
    ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        loss, x_recon, x_data, perplexity = x
        reconstruction_loss = F.mse_loss(x_recon, x_data) / self._data_variance

        return loss, reconstruction_loss, perplexity

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, x, perplexity

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch

        vq_loss, x_recon, _, perplexity = self.forward(x)
        _, reconstruction_loss, _ = self.criterion((vq_loss, x_recon, x, perplexity))

        loss = vq_loss + reconstruction_loss

        # LOGGING
        self.log_with_postfix(
            f"train/loss",
            loss,
        )
        self.log_with_postfix(
            f"train/vq_loss",
            vq_loss,
        )
        self.log_with_postfix(
            f"train/reconstruction_loss",
            reconstruction_loss,
        )
        self.log_with_postfix(
            f"train/perplexity",
            perplexity,
        )

        return {
            "loss": loss,
            "forward_output": (vq_loss, x_recon, x, perplexity),
        }

    def validation_step(self, batch, batch_idx):
        x, y, *_ = batch

        vq_loss, x_recon, _, perplexity = self.forward(x)
        _, reconstruction_loss, _ = self.criterion((vq_loss, x_recon, x, perplexity))

        loss = vq_loss + reconstruction_loss

        # LOGGING
        self.log_with_postfix(
            f"val/loss",
            loss,
        )
        self.log_with_postfix(
            f"val/vq_loss",
            vq_loss,
        )
        self.log_with_postfix(
            f"val/reconstruction_loss",
            reconstruction_loss,
        )
        self.log_with_postfix(
            f"val/perplexity",
            perplexity,
        )

        return {
            "loss": loss,
            "forward_output": (vq_loss, x_recon, x, perplexity),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        return optimizer

    def log_with_postfix(self, name: str, value: t.Any, *args, **kwargs):
        self.log_dict(
            {
                f"{name}/experience_step_{self.experience_step}": value,
            },
            *args,
            **kwargs,
        )
