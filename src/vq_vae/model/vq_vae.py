import typing as t
from itertools import chain

import lpips
import torch
from torch import nn
from torch.nn import functional as F

from src.avalanche.model.cl_model import CLModel
from src.vq_vae.model.decoder import Decoder
from src.vq_vae.model.encoder import Encoder
from src.vq_vae.model.quiantizer import VectorQuantizer, VectorQuantizerEMA
from src.vq_vae.model.resnet import ResidualStack


class ForwardOutput(t.NamedTuple):
    vq_loss: torch.Tensor
    x_data: torch.Tensor
    x_recon: torch.Tensor
    quantized: torch.Tensor
    perplexity: torch.Tensor
    clf_logits: torch.Tensor


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
        num_classes,
        embedding_dim,
        commitment_cost,
        decay=0,
        data_variance=1,
        learning_rate: float = 1e-3,
        regularization_dropout: float = 0.0,
        regularization_lambda: float = 0.0,
        use_lpips: bool = False,
        vq_loss_weight: float = 1,
        reconstruction_loss_weight: float = 1,
        downstream_loss_weight: float = 1,
        cnn_clf: bool = True,
    ):
        super().__init__()

        self._downstream_loss_weight = downstream_loss_weight
        self._reconstruction_loss_weight = reconstruction_loss_weight
        self._vq_loss_weight = vq_loss_weight

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

        if cnn_clf:
            self.clf_head = nn.Sequential(
                ResidualStack(
                    in_channels=embedding_dim,
                    num_hiddens=embedding_dim,
                    num_residual_layers=1,
                    num_residual_hiddens=32,
                    regularization_dropout=0,
                ),
                nn.Flatten(),
                nn.Linear(8 * 8 * embedding_dim, num_classes),
            )
        else:
            self.clf_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(8 * 8 * embedding_dim, num_classes),
            )

        if use_lpips:
            self._lpips = lpips.LPIPS(net="vgg")
            self.reconstruction_loss_fn = (
                lambda x, y: self._lpips(x, y).mean()
                + F.l1_loss(x, y, reduction="mean") / self._data_variance
            )
        else:
            self.reconstruction_loss_fn = (
                lambda x, y: F.l1_loss(x, y, reduction="mean") / self._data_variance
            )

    def criterion(self, x: ForwardOutput, y) -> CriterionOutput:
        x_recon = x.x_recon
        x_data = x.x_data
        logits = x.clf_logits

        reconstruction_loss = self.reconstruction_loss_fn(x_recon, x_data)

        past_labels = y == -1
        current_labels = torch.isin(
            y, torch.tensor([-1, -2], device=self.device), invert=True
        )

        cel_current_labels = F.cross_entropy(logits[current_labels], y[current_labels])
        neg_entropy_past_labels = 0

        if past_labels.any():
            neg_entropy_past_labels = F.softmax(
                logits[past_labels], dim=1
            ) * F.log_softmax(logits[past_labels], dim=1)
            neg_entropy_past_labels = -1.0 * neg_entropy_past_labels.mean()

        clf_acc = (logits.argmax(dim=-1) == y).float().mean()
        clf_loss = neg_entropy_past_labels + cel_current_labels

        return CriterionOutput(
            vq_loss=x.vq_loss,
            reconstruction_loss=reconstruction_loss,
            clf_loss=clf_loss,
            clf_acc=clf_acc,
            perplexity=x.perplexity,
        )

    def forward(self, x) -> ForwardOutput:
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        loss, quantized, perplexity, _ = self.vq_vae(z)

        x_recon = self.decoder(quantized)
        logits = self.clf_head(quantized)

        return ForwardOutput(
            vq_loss=loss,
            x_recon=x_recon,
            x_data=x,
            quantized=quantized,
            perplexity=perplexity,
            clf_logits=logits,
        )

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch

        forward_output: ForwardOutput = self.forward(x)
        criterion_output: CriterionOutput = self.criterion(forward_output, y)

        loss = (
            self._vq_loss_weight * criterion_output.vq_loss
            + self._reconstruction_loss_weight * criterion_output.reconstruction_loss
            + self._downstream_loss_weight * criterion_output.clf_loss
        )

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
        self.log_with_postfix(
            f"train/clf_loss",
            criterion_output.clf_loss,
        )
        self.log_with_postfix(
            f"train/clf_accuracy",
            criterion_output.clf_acc,
        )

        return {
            "loss": loss,
            "forward_output": forward_output,
        }

    def validation_step(self, batch, batch_idx):
        x, y, *_ = batch

        forward_output = self.forward(x)
        criterion_output = self.criterion(forward_output, y)

        loss = (
            self._vq_loss_weight * criterion_output.vq_loss
            + self._reconstruction_loss_weight * criterion_output.reconstruction_loss
            + self._downstream_loss_weight * criterion_output.clf_loss
        )

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
        self.log_with_postfix(
            f"val/clf_loss",
            criterion_output.clf_loss,
        )
        self.log_with_postfix(
            f"val/clf_accuracy",
            criterion_output.clf_acc,
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
                self.clf_head.parameters(),
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
