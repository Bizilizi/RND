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
    clf_logits: torch.Tensor
    lm_logits: torch.Tensor
    encoding_indices: torch.Tensor


class CriterionOutput(t.NamedTuple):
    vq_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    lm_loss: torch.Tensor
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
        self._latent_sos_token = num_embeddings + 1

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

    def criterion(self, forward_output: ForwardOutput, y) -> CriterionOutput:
        reconstruction_loss = F.mse_loss(forward_output.x_recon, forward_output.x_data)
        contrastive_loss = self.c_loss(forward_output.image_emb, y)

        # Compute accuracy if classification head presents
        if forward_output.clf_logits is not None:
            clf_loss = F.cross_entropy(forward_output.clf_logits, y)
            clf_acc = (forward_output.clf_logits.argmax(dim=-1) == y).float().mean()
        else:
            clf_loss = clf_acc = None

        # Compute lm loss
        if forward_output.lm_logits is not None:
            lm_logits = forward_output.lm_logits
            lm_loss = F.cross_entropy(
                lm_logits[:, :-1].reshape(-1, lm_logits.shape[-1]),
                forward_output.encoding_indices.reshape(-1),
            )
        else:
            lm_loss = None

        return CriterionOutput(
            vq_loss=forward_output.vq_loss,
            reconstruction_loss=reconstruction_loss,
            lm_loss=lm_loss,
            clf_loss=clf_loss,
            clf_acc=clf_acc,
            contrastive_loss=contrastive_loss,
            perplexity=forward_output.perplexity,
        )

    def forward(self, x) -> ForwardOutput:
        z = self.encoder(x)
        image_emb = z[:, 0]
        patches_emb = z[:, 1:]

        vq_loss, quantized, perplexity, encoding_indices = self.vq_vae(patches_emb)
        x_recon, lm_logits = self.decoder(quantized)

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
            lm_logits=lm_logits,
            encoding_indices=encoding_indices,
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
        self.log_with_postfix(
            f"train/contrastive_loss",
            criterion_output.contrastive_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"train/pixels_lm_loss",
            criterion_output.lm_loss.cpu().item(),
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
        self.log_with_postfix(
            f"val/contrastive_loss",
            criterion_output.contrastive_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"val/pixels_lm_loss",
            criterion_output.lm_loss.cpu().item(),
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
