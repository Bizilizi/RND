import math
import typing as t
from itertools import chain

import lpips
import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ContrastiveLoss
from torch import nn
from torch.nn import functional as F

from src.avalanche.model.cl_model import CLModel
from src.transformer_vq_vae.model.decoder import MAEDecoder
from src.transformer_vq_vae.model.encoder import MAEEncoder
from src.transformer_vq_vae.model.quiantizer import VitVectorQuantizerEMA

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
    mask: torch.Tensor


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
        num_class_embeddings,
        embedding_dim,
        commitment_cost,
        decay=0,
        learning_rate: float = 1e-3,
        weight_decay=0.005,
        image_size=32,
        patch_size=2,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
        mask_ratio=0.75,
        use_lpips: bool = True,
    ) -> None:
        super().__init__()

        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._embedding_dim = embedding_dim
        self._latent_sos_token = num_embeddings + 1
        self._mask_ratio = mask_ratio

        self.encoder = MAEEncoder(
            image_size,
            patch_size,
            embedding_dim,
            encoder_layer,
            encoder_head,
        )
        self.feature_quantization = VitVectorQuantizerEMA(
            num_embeddings, embedding_dim, commitment_cost, decay
        )
        self.class_quantization = VitVectorQuantizerEMA(
            num_class_embeddings, embedding_dim, commitment_cost, decay
        )
        self.decoder = MAEDecoder(
            image_size, patch_size, embedding_dim, decoder_layer, decoder_head
        )

        self.clf_head = None
        self.experience_step = 0

        if use_lpips:
            self._lpips = lpips.LPIPS(net="vgg")
            self.rec_loss_fn = lambda x, y: (
                self._lpips(x, y).mean()
                + F.l1_loss(x, y, reduction="mean") / self._data_variance
            )
        else:
            self.rec_loss_fn = (
                lambda x, y: F.l1_loss(x, y, reduction="mean") / self._data_variance
            )

        self._data_variance = 0.06328692405746414

    def set_clf_head(self, model: "CnnClassifier"):
        self.__dict__["clf_head"] = model

    def reset_clf_head(self):
        self.clf_head = None

    def criterion(self, forward_output: ForwardOutput, y) -> CriterionOutput:
        x_recon = forward_output.x_recon
        x_data = forward_output.x_data

        # Compute contrastive loss
        reconstruction_loss = self.rec_loss_fn(x_recon, x_data)

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
        # Extract features from backbone
        features, backward_indexes = self.encoder(x)
        image_emb = features[0]

        (
            feature_vq_loss,
            quantized_features,
            feature_perplexity,
            _,
        ) = self.feature_quantization(features[1:])
        (
            class_vq_loss,
            quantized_class_emb,
            class_perplexity,
            _,
        ) = self.class_quantization(image_emb[None])

        quantized_features = torch.cat([quantized_class_emb, quantized_features])
        x_recon, mask = self.decoder(quantized_features, backward_indexes)

        # If the model has classification head
        # we calculate image embedding based on output of the encoder
        # without masking random patches
        clf_logits = None
        if self.clf_head is not None:
            non_shuffled_features, _ = self.encoder(x, shuffle=False)
            image_emb = non_shuffled_features[0]
            clf_logits = self.clf_head(image_emb)

        return ForwardOutput(
            vq_loss=feature_vq_loss + class_vq_loss,
            x_recon=x_recon,
            x_data=x,
            quantized=quantized_features,
            perplexity=feature_perplexity + class_perplexity,
            image_emb=image_emb,
            clf_logits=clf_logits,
            mask=mask,
        )

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch

        forward_output = self.forward(x)
        criterion_output = self.criterion(forward_output, y)

        loss = criterion_output.vq_loss + criterion_output.reconstruction_loss

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

        loss = criterion_output.vq_loss + criterion_output.reconstruction_loss

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

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            chain(
                self.encoder.parameters(),
                self.feature_quantization.parameters(),
                self.decoder.parameters(),
            ),
            lr=self._learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self._weight_decay,
        )

        lr_func = lambda epoch: min(
            (epoch + 1) / (200 + 1e-8),
            0.5 * (math.cos(epoch / 2000 * math.pi) + 1),
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_func, verbose=True
        )

        return [optimizer], [lr_scheduler]

    def log_with_postfix(self, name: str, value: t.Any, *args, **kwargs):
        self.log_dict(
            {
                f"{name}/experience_step_{self.experience_step}": value,
            },
            *args,
            **kwargs,
        )
