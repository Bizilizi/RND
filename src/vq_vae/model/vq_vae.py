import typing as t
from itertools import chain

import lpips
import torch
from torch import nn
from torch.nn import functional as F

from src.avalanche.model.cl_model import CLModel
from src.vq_vae.model.decoder import Decoder
from src.vq_vae.model.encoder import Encoder
from src.vq_vae.model.quiantizer import (
    VectorQuantizer,
    VectorQuantizerEMA,
)

if t.TYPE_CHECKING:
    from src.vq_vae.model.classification_head import CnnClassifier


class ForwardOutput(t.NamedTuple):
    vq_loss: torch.Tensor
    x_data: torch.Tensor
    x_recon: torch.Tensor
    quantized: torch.Tensor
    perplexity: torch.Tensor
    logits: torch.Tensor


class CriterionOutput(t.NamedTuple):
    vq_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    clf_loss: torch.Tensor
    clf_acc: torch.Tensor
    perplexity: torch.Tensor


class LinearBottleneck(nn.Module):
    def __init__(self, width, height, hidden_dim):
        super().__init__()

        self.width = width
        self.height = height
        self.hidden_dim = hidden_dim

        self.module = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(width * height * hidden_dim)
        )

    def forward(self, x):
        x = self.module(x)
        x = x.reshape(-1, self.height, self.width, self.hidden_dim)

        return x


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
        use_lpips: bool = True,
    ):
        super().__init__()

        self._data_variance = data_variance
        self._learning_rate = learning_rate
        self._weight_decay = regularization_lambda
        self._regularization_dropout = regularization_dropout
        self._embedding_dim = embedding_dim

        self.encoder = Encoder(
            in_channels=3,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            regularization_dropout=regularization_dropout,
        )
        self.pre_vq_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_hiddens,
                out_channels=embedding_dim,
                kernel_size=1,
                stride=1,
            ),
            # LinearBottleneck(8, 8, embedding_dim),
        )
        if decay > 0.0:
            self.vq_vae = VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, decay
            )
        else:
            self.vq_vae = VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )
        self.decoder = nn.Sequential(
            # LinearBottleneck(8, 8, embedding_dim),
            Decoder(
                in_channels=embedding_dim,
                num_hiddens=num_hiddens,
                num_residual_layers=num_residual_layers,
                num_residual_hiddens=num_residual_hiddens,
                regularization_dropout=regularization_dropout,
            ),
        )

        self.clf_head = None

        if use_lpips:
            self._lpips = lpips.LPIPS(net="vgg")
            self.reconstruction_loss_fn = lambda x, y: self._lpips(x, y).mean()
        else:
            self.reconstruction_loss_fn = (
                lambda x, y: F.mse_loss(x, y, reduction="mean") / self._data_variance
            )

    def set_clf_head(self, model: "CnnClassifier"):
        self.__dict__["clf_head"] = model

    def reset_clf_head(self):
        self.clf_head = None

    def criterion(self, x, y):
        loss, x_recon, quantized, x_data, perplexity, logits = x
        reconstruction_loss = self.reconstruction_loss_fn(x_recon, x_data)

        if logits is not None:
            clf_loss = F.cross_entropy(logits, y)
            clf_acc = (logits.argmax(dim=-1) == y).float().mean()
        else:
            clf_loss = clf_acc = None

        return CriterionOutput(
            vq_loss=loss,
            reconstruction_loss=reconstruction_loss,
            clf_loss=clf_loss,
            clf_acc=clf_acc,
            perplexity=perplexity,
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        loss, quantized, perplexity, _ = self.vq_vae(z)
        x_recon = self.decoder(quantized)

        if self.clf_head is not None:
            logits = self.clf_head(quantized)
        else:
            logits = None

        return ForwardOutput(
            vq_loss=loss,
            x_recon=x_recon,
            x_data=x,
            quantized=quantized,
            perplexity=perplexity,
            logits=logits,
        )

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch

        forward_output = self.forward(x)
        criterion_output = self.criterion(forward_output, y)

        loss = criterion_output.vq_loss + criterion_output.reconstruction_loss

        # LOGGING
        self.log_with_postfix(
            f"train/loss",
            loss,
        )
        self.log_with_postfix(
            f"train/vq_loss",
            criterion_output.vq_loss,
        )
        self.log_with_postfix(
            f"train/reconstruction_loss",
            criterion_output.reconstruction_loss,
        )
        self.log_with_postfix(
            f"train/perplexity",
            criterion_output.perplexity,
        )

        return {
            "loss": loss,
            "forward_output": forward_output,
        }

    def validation_step(self, batch, batch_idx):
        x, y, *_ = batch

        forward_output = self.forward(x)
        criterion_output = self.criterion(forward_output, y)

        loss = criterion_output.vq_loss + criterion_output.reconstruction_loss

        # LOGGING
        self.log_with_postfix(
            f"val/loss",
            loss,
        )
        self.log_with_postfix(
            f"val/vq_loss",
            criterion_output.vq_loss,
        )
        self.log_with_postfix(
            f"val/reconstruction_loss",
            criterion_output.reconstruction_loss,
        )
        self.log_with_postfix(
            f"val/perplexity",
            criterion_output.perplexity,
        )

        return {
            "loss": loss,
            "forward_output": forward_output,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            chain(
                self.encoder.parameters(),
                self.pre_vq_conv.parameters(),
                self.vq_vae.parameters(),
                self.decoder.parameters(),
            ),
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
