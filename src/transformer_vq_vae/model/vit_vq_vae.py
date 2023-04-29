import math
import typing as t
from itertools import chain

import torch
from torch import nn
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ContrastiveLoss
from torch.nn import functional as F

from src.avalanche.model.cl_model import CLModel
from src.transformer_vq_vae.model.decoder import GPTDecoder, VITDecoder
from src.transformer_vq_vae.model.encoder import VitEncoder
from src.transformer_vq_vae.model.quiantizer import (
    VitVectorQuantizerEMA,
)

if t.TYPE_CHECKING:
    from src.transformer_vq_vae.model.classification_head import CnnClassifier


class ForwardOutput(t.NamedTuple):
    vq_loss: torch.Tensor

    x_data: torch.Tensor
    x_recon: torch.Tensor

    quantized: torch.Tensor
    perplexity: torch.Tensor

    image_emb: torch.Tensor
    clf_logits: torch.Tensor
    encoding_indices: torch.Tensor
    non_masked_indices: torch.Tensor


class CriterionOutput(t.NamedTuple):
    vq_loss: torch.Tensor
    reconstruction_loss: torch.Tensor

    clf_loss: torch.Tensor
    clf_acc: torch.Tensor
    perplexity: torch.Tensor


class VitVQVae(CLModel):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay=0,
        data_variance=1,
        learning_rate: float = 1e-3,
        embeddings_distance: str = "cosine",
        patch_size=4,
        patch_corruption_rate: float = 0.2,
        vq_loss_weight: float = 1,
        reconstruction_loss_weight: float = 1,
        contrastive_loss_loss_weight: float = 1,
        encoder_mlm_loss_loss_weight: float = 1,
        decoder_regression_loss_loss_weight: float = 1,
    ):
        super().__init__()

        # loss weights
        self.decoder_regression_loss_loss_weight = decoder_regression_loss_loss_weight
        self.encoder_mlm_loss_loss_weight = encoder_mlm_loss_loss_weight
        self.contrastive_loss_loss_weight = contrastive_loss_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight

        self.vq_loss_weight = vq_loss_weight
        self._data_variance = data_variance
        self._learning_rate = learning_rate
        self._embedding_dim = embedding_dim
        self._latent_sos_token = num_embeddings + 1

        self.encoder = VitEncoder(
            embeddings_dim=embedding_dim,
            patch_size=patch_size,
            corruption_rate=patch_corruption_rate,
        )
        self.vq_vae = VitVectorQuantizerEMA(
            num_embeddings, embedding_dim, commitment_cost, decay
        )
        self.masked_patch_token = nn.Embedding(1, embedding_dim=embedding_dim)
        self.decoder = VITDecoder(
            embedding_dim, num_embeddings, n_positions=8 * 8, patch_size=patch_size
        )

        if embeddings_distance == "cosine":
            self.c_loss = ContrastiveLoss(
                pos_margin=1, neg_margin=0, distance=CosineSimilarity()
            )
        else:
            self.c_loss = ContrastiveLoss()

        self.clf_head = None
        self.experience_step = 0

    def set_clf_head(self, model: "CnnClassifier"):
        self.__dict__["clf_head"] = model

    def reset_clf_head(self):
        self.clf_head = None

    def criterion(self, forward_output: ForwardOutput, y) -> CriterionOutput:
        x_recon = forward_output.x_recon
        x_data = forward_output.x_data

        # Compute contrastive loss
        reconstruction_loss = (
            F.l1_loss(x_recon, x_data, reduction="mean") / self._data_variance
        )

        # Compute accuracy if classification head presents
        clf_loss = clf_acc = torch.tensor(0, device=self.device)
        if forward_output.clf_logits is not None:
            clf_loss = F.cross_entropy(forward_output.clf_logits, y)
            clf_acc = (forward_output.clf_logits.argmax(dim=-1) == y).float().mean()

        return CriterionOutput(
            vq_loss=forward_output.vq_loss,
            reconstruction_loss=reconstruction_loss,
            clf_loss=clf_loss,
            clf_acc=clf_acc,
            perplexity=forward_output.perplexity,
        )

    def forward(self, x) -> ForwardOutput:
        B, nc, w, h = x.shape
        num_patches = self.encoder.base_vit.patch_embed.num_patches

        # Apply encoder twice (clean and corrupted inputs)
        features, non_masked_indices = self.encoder(x)

        # Extract image embedding from uncorrupted input
        image_emb = features[:, 0]
        patches_emb = features[:, 1:]

        # Quantize input
        vq_loss, quantized_patches, perplexity, encoding_indices = self.vq_vae(
            patches_emb
        )

        quantized = self.masked_patch_token(
            torch.zeros(
                x.shape[0] * num_patches,
                device=self.device,
            ).long()
        )
        quantized[non_masked_indices] = quantized_patches.flatten(0, 1)
        quantized = quantized.reshape(B, num_patches, -1)

        x_recon, _ = self.decoder(quantized)

        if self.clf_head is not None:
            clf_logits = self.clf_head(image_emb)
        else:
            clf_logits = None

        return ForwardOutput(
            vq_loss=vq_loss,
            x_recon=x_recon,
            x_data=x,
            quantized=quantized,
            perplexity=perplexity,
            image_emb=image_emb,
            clf_logits=clf_logits,
            encoding_indices=encoding_indices,
            non_masked_indices=non_masked_indices,
        )

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch

        forward_output = self.forward(x)
        criterion_output = self.criterion(forward_output, y)

        loss = (
            criterion_output.vq_loss * self.vq_loss_weight
            + criterion_output.reconstruction_loss * self.reconstruction_loss_weight
        )

        # LOGGING
        self.log_with_postfix(
            f"train/loss",
            loss.cpu().item(),
        )
        self.log_with_postfix(
            f"train/vq_loss",
            criterion_output.vq_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"train/reconstruction_loss",
            criterion_output.reconstruction_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"train/perplexity",
            criterion_output.perplexity.cpu().item(),
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
            criterion_output.vq_loss * self.vq_loss_weight
            + criterion_output.reconstruction_loss * self.reconstruction_loss_weight
        )

        # LOGGING
        self.log_with_postfix(
            f"val/loss",
            loss.cpu().item(),
        )
        self.log_with_postfix(
            f"val/vq_loss",
            criterion_output.vq_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"val/reconstruction_loss",
            criterion_output.reconstruction_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"val/perplexity",
            criterion_output.perplexity.cpu().item(),
        )

        return {
            "loss": loss,
            "forward_output": forward_output,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            chain(
                self.encoder.parameters(),
                self.vq_vae.parameters(),
                self.decoder.parameters(),
            ),
            lr=self._learning_rate,
        )

        lr_func = lambda epoch: min(
            (epoch + 1) / (200 + 1e-8),
            0.5 * (math.cos(epoch / 2000 * math.pi) + 1),
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_func, verbose=True
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
