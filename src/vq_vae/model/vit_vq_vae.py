import typing as t
from itertools import chain

import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ContrastiveLoss
from torch.nn import functional as F

from src.avalanche.model.cl_model import CLModel
from src.vq_vae.model.decoder import GPTDecoder
from src.vq_vae.model.encoder import VitEncoder
from src.vq_vae.model.quiantizer import (
    VitVectorQuantizerEMA,
)

if t.TYPE_CHECKING:
    from src.vq_vae.model.classification_head import CnnClassifier


class ForwardOutput(t.NamedTuple):
    vq_loss: torch.Tensor
    x_data: torch.Tensor
    x_recon: torch.Tensor
    quantized: torch.Tensor
    perplexity: torch.Tensor
    image_emb: torch.Tensor
    logits: torch.Tensor


class CriterionOutput(t.NamedTuple):
    vq_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    clf_loss: torch.Tensor
    clf_acc: torch.Tensor
    contrastive_loss: torch.Tensor
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
    ):
        super().__init__()

        self._data_variance = data_variance
        self._learning_rate = learning_rate
        self._embedding_dim = embedding_dim

        self.encoder = VitEncoder(embeddings_dim=embedding_dim, patch_size=patch_size)

        self.vq_vae = VitVectorQuantizerEMA(
            num_embeddings, embedding_dim, commitment_cost, decay
        )

        self.decoder = GPTDecoder(
            embedding_dim, num_embeddings, n_positions=8 * 8, patch_size=patch_size
        )

        if embeddings_distance == "cosine":
            self.c_loss = ContrastiveLoss(
                pos_margin=1, neg_margin=0, distance=CosineSimilarity()
            )
        else:
            self.c_loss = ContrastiveLoss()

        self.clf_head = None

    def set_clf_head(self, model: "CnnClassifier"):
        self.__dict__["clf_head"] = model

    def reset_clf_head(self):
        self.clf_head = None

    def criterion(self, x, y):
        vq_loss, x_recon, quantized, x_data, perplexity, image_emb, logits = x
        reconstruction_loss = F.mse_loss(x_recon, x_data)
        contrastive_loss = self.c_loss(image_emb, y)

        if logits is not None:
            clf_loss = F.cross_entropy(logits, y)
            clf_acc = (logits.argmax(dim=-1) == y).float().mean()
        else:
            clf_loss = clf_acc = None

        return CriterionOutput(
            vq_loss=vq_loss,
            reconstruction_loss=reconstruction_loss,
            clf_loss=clf_loss,
            clf_acc=clf_acc,
            contrastive_loss=contrastive_loss,
            perplexity=perplexity,
        )

    def forward(self, x):
        z = self.encoder(x)
        image_emb = z[:, 0]
        patches_emb = z[:, 1:]

        vq_loss, quantized, perplexity, _ = self.vq_vae(patches_emb)
        x_recon, _ = self.decoder(quantized)

        if self.clf_head is not None:
            logits = self.clf_head(image_emb)
        else:
            logits = None

        return ForwardOutput(
            vq_loss=vq_loss,
            x_recon=x_recon,
            x_data=x,
            quantized=quantized,
            perplexity=perplexity,
            image_emb=image_emb,
            logits=logits,
        )

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch

        forward_output = self.forward(x)
        criterion_output = self.criterion(forward_output, y)

        loss = (
            criterion_output.vq_loss
            + criterion_output.reconstruction_loss
            + criterion_output.contrastive_loss
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
            f"train/contrastive_loss",
            criterion_output.contrastive_loss,
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
            criterion_output.vq_loss
            + criterion_output.reconstruction_loss
            + criterion_output.contrastive_loss
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
            f"val/contrastive_loss",
            criterion_output.contrastive_loss,
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
        return optimizer

    def log_with_postfix(self, name: str, value: t.Any, *args, **kwargs):
        self.log_dict(
            {
                f"{name}/experience_step_{self.experience_step}": value,
            },
            *args,
            **kwargs,
        )
