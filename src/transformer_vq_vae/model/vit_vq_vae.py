import dataclasses

import math
import typing as t
from itertools import chain

import lpips
import torch
from torch.nn import functional as F

from src.avalanche.model.cl_model import CLModel
from src.transformer_vq_vae.model.decoder import MAEDecoder
from src.transformer_vq_vae.model.encoder import MAEEncoder
from src.transformer_vq_vae.model.quiantizer import (
    FeatureQuantizer,
)

if t.TYPE_CHECKING:
    from src.transformer_vq_vae.model.classification_head import EmbClassifier


@dataclasses.dataclass
class ForwardOutput:
    vq_loss: torch.Tensor

    x_data: torch.Tensor
    x_recon: torch.Tensor
    x_indices: torch.Tensor

    quantized: torch.Tensor
    latent_distances: torch.Tensor
    perplexity: torch.Tensor

    image_emb: torch.Tensor
    clf_logits: torch.Tensor
    mask: torch.Tensor


@dataclasses.dataclass
class CriterionOutput:
    vq_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    cycle_consistency_loss: torch.Tensor

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
        mask_token_id: int,
        num_epochs: int,
        batch_size: int,
        decay=0,
        learning_rate: float = 1e-3,
        weight_decay=0.05,
        image_size=32,
        patch_size=2,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
        mask_ratio=0.75,
        use_lpips: bool = True,
        cycle_consistency_power=3,
        cycle_consistency_weight=1,
        cycle_consistency_sigma: float = 1,
        current_samples_loss_weight=2,
        past_samples_loss_weight=1,
        precision: str = "32-true",
        accelerator: str = "cuda",
        quantize_features: bool = True,
        data_variance: float = 0.06328692405746414,
    ) -> None:
        super().__init__()

        self._num_embeddings = num_embeddings
        self._num_class_embeddings = num_class_embeddings
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._embedding_dim = embedding_dim
        self._latent_sos_token = num_embeddings + 1
        self._mask_ratio = mask_ratio
        self._mask_token_id = mask_token_id
        self._precision_dtype = torch.half if precision == "16-mixed" else torch.float32
        self._accelerator = accelerator
        self._current_samples_loss_weight = current_samples_loss_weight
        self._past_samples_loss_weight = past_samples_loss_weight
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._cycle_consistency_sigma = cycle_consistency_sigma
        self._quantize_features = quantize_features

        self.encoder = MAEEncoder(
            image_size,
            patch_size,
            embedding_dim,
            encoder_layer,
            encoder_head,
        )
        self.feature_quantization = FeatureQuantizer(
            num_class_embeddings, num_embeddings, embedding_dim, commitment_cost, decay
        )
        self.decoder = MAEDecoder(
            image_size, patch_size, embedding_dim, decoder_layer, decoder_head
        )

        self.clf_head = None
        self.experience_step = 0
        self.cycle_consistency_power = cycle_consistency_power
        self.cycle_consistency_weight = cycle_consistency_weight
        self.use_lpips = use_lpips
        self._data_variance = data_variance

        if self.use_lpips:
            self._lpips = lpips.LPIPS(net="vgg")

    def get_reconstruction_loss(
        self, x: torch.Tensor, x_rec: torch.Tensor, y: torch.Tensor
    ):
        # Create weight vector to shift gradient towards current dataset
        weight_tensor = torch.ones(y.shape[0], device=self.device)
        weight_tensor[y >= 0] = self._current_samples_loss_weight

        if self.use_lpips:
            lpips_loss = (self._lpips(x, x_rec) * weight_tensor).mean()
            l1_loss = torch.mean(
                F.l1_loss(x, x_rec, reduction="none").mean((1, 2, 3))
                * weight_tensor
                / self._data_variance
            )
            reconstruction_loss = lpips_loss + l1_loss

        else:
            reconstruction_loss = torch.mean(
                F.l1_loss(x, x_rec, reduction="none").mean((1, 2, 3))
                * weight_tensor
                / self._data_variance
            )

        return reconstruction_loss

    def set_clf_head(self, model: "EmbClassifier"):
        self.__dict__["clf_head"] = model

    def reset_clf_head(self):
        self.clf_head = None

    def criterion(self, forward_output: ForwardOutput, y) -> CriterionOutput:
        # prepare default values
        clf_loss = clf_acc = torch.tensor(0.0, device=self.device)
        cycle_consistency_loss = torch.tensor(0.0, device=self.device)
        reconstruction_loss = torch.tensor(0.0, device=self.device)

        # unpack variables from forward output
        x_recon = forward_output.x_recon
        x_data = forward_output.x_data
        x_indices = forward_output.x_indices
        latent_distances = forward_output.latent_distances

        past_data = y == -1
        current_data = y >= 0

        # Compute reconstruction loss
        if current_data.any():
            reconstruction_loss = self.get_reconstruction_loss(
                x_recon[current_data], x_data[current_data], y[current_data]
            )

        if self._past_samples_loss_weight != 0 and past_data.any():
            reconstruction_loss += self.get_reconstruction_loss(
                x_recon[past_data], x_data[past_data], y[past_data]
            )

        # Compute accuracy if classification head presents
        if forward_output.clf_logits is not None:
            clf_loss = F.cross_entropy(forward_output.clf_logits, y)
            clf_acc = (forward_output.clf_logits.argmax(dim=-1) == y).float().mean()

        # Compute consistency loss
        if (
            latent_distances is not None
            and self.cycle_consistency_weight != 0
            and past_data.any()
        ):
            distances = forward_output.latent_distances[past_data]
            indices = x_indices[past_data].long()

            q_logits = -1 / 2 * distances / self._cycle_consistency_sigma

            q_logits = q_logits.flatten(0, 1)
            q_indices = indices.flatten()

            # Remove loss for mask token
            q_logits = q_logits[q_indices != self._mask_token_id]
            q_indices = q_indices[q_indices != self._mask_token_id]

            cycle_consistency_loss = F.cross_entropy(q_logits, q_indices)

        return CriterionOutput(
            vq_loss=forward_output.vq_loss,
            reconstruction_loss=reconstruction_loss,
            cycle_consistency_loss=cycle_consistency_loss,
            clf_loss=clf_loss,
            clf_acc=clf_acc,
            perplexity=forward_output.perplexity,
        )

    def forward(self, x) -> ForwardOutput:
        # prepare default values
        latent_distances = None
        perplexity = torch.tensor(0.0, device=self.device)
        vq_loss = torch.tensor(0.0, device=self.device)
        clf_logits = None
        image_emb = None

        # Extract features from backbone
        masked_features, full_features, backward_indexes = self.encoder(
            x, return_full_features=True
        )

        if self._quantize_features:
            with torch.autocast(self._accelerator, dtype=torch.float32):
                (
                    vq_loss,
                    masked_features,
                    perplexity,
                    *_,
                ) = self.feature_quantization(masked_features)
                (*_, latent_distances) = self.feature_quantization(
                    full_features, return_distances=True
                )

        x_recon, mask = self.decoder(masked_features, backward_indexes)

        # If the model has classification head
        # we calculate image embedding based on output of the encoder
        # without masking random patches
        if self.clf_head is not None:
            image_emb = full_features[0]
            clf_logits = self.clf_head(image_emb)

        return ForwardOutput(
            vq_loss=vq_loss,
            x_recon=x_recon,
            x_data=x,
            x_indices=None,
            quantized=masked_features,
            perplexity=perplexity,
            image_emb=image_emb,
            clf_logits=clf_logits,
            mask=mask,
            latent_distances=latent_distances,
        )

    def training_step(self, batch, batch_idx):
        data, y, *_ = batch

        x = data["images"]

        forward_output = self.forward(x)
        forward_output.x_data = x
        forward_output.x_indices = data["indices"]

        criterion_output = self.criterion(forward_output, y)

        loss = (
            criterion_output.vq_loss
            + criterion_output.reconstruction_loss
            + criterion_output.cycle_consistency_loss * self.cycle_consistency_weight
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
            f"train/cycle_consistency_loss",
            criterion_output.cycle_consistency_loss.cpu().item(),
        )

        return {
            "loss": loss,
            "forward_output": forward_output,
        }

    def validation_step(self, batch, batch_idx):
        data, y, *_ = batch

        x = data["images"]

        forward_output = self.forward(x)
        forward_output.x_data = x
        forward_output.x_indices = data["indices"]

        criterion_output = self.criterion(forward_output, y)

        loss = (
            criterion_output.vq_loss
            + criterion_output.reconstruction_loss
            + criterion_output.cycle_consistency_loss * self.cycle_consistency_weight
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
            f"val/cycle_consistency_loss",
            criterion_output.cycle_consistency_loss.cpu().item(),
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
                self.decoder.parameters(),
            ),
            lr=self._learning_rate * self._batch_size / 256,
            betas=(0.9, 0.95),
            weight_decay=self._weight_decay,
        )

        warmup = min(200, self._num_epochs // 3)
        lr_func = lambda epoch: min(
            (epoch + 1) / (warmup + 1e-8),
            0.5 * (math.cos(epoch / self._num_epochs * math.pi) + 1),
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_func, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def log_with_postfix(self, name: str, value: t.Any, *args, **kwargs):
        self.log_dict(
            {
                f"{name}/experience_step_{self.experience_step}": value,
            },
            sync_dist=True,
            *args,
            **kwargs,
        )

    def unfreeze(self) -> None:
        super().unfreeze()

        for param in self.feature_quantization.parameters():
            param.requires_grad = False
