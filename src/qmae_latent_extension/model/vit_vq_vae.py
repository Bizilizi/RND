import dataclasses

import math
import typing as t
from itertools import chain

import lpips
import torch
from einops import rearrange
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ContrastiveLoss, TripletMarginLoss
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn import functional as F

from src.avalanche.model.cl_model import CLModel
from src.qmae_latent_extension.model.decoder import MAEDecoder
from src.qmae_latent_extension.model.encoder import MAEEncoder
from src.qmae_latent_extension.model.quiantizer import (
    VectorQuantizerEMA,
)

if t.TYPE_CHECKING:
    from src.qmae_latent_extension.model.classification_head import EmbClassifier


@dataclasses.dataclass
class ForwardOutput:
    vq_loss: torch.Tensor

    x_data: torch.Tensor
    x_recon: torch.Tensor
    x_indices: torch.Tensor

    quantized: torch.Tensor
    latent_distances: torch.Tensor
    second_order_distances: torch.Tensor
    perplexity: torch.Tensor
    avg_probs: torch.Tensor

    image_emb: torch.Tensor
    clf_logits: torch.Tensor
    mask: torch.Tensor

    past_data_mask: torch.Tensor


@dataclasses.dataclass
class CriterionOutput:
    vq_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    past_cycle_consistency_loss: torch.Tensor
    current_cycle_consistency_loss: torch.Tensor
    triplet_loss: torch.Tensor

    clf_loss: torch.Tensor
    clf_acc: torch.Tensor
    perplexity: torch.Tensor


class VitVQVae(CLModel):
    def __init__(
        self,
        num_embeddings,
        num_embeddings_per_step,
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
        self.feature_quantization = VectorQuantizerEMA(
            num_embeddings,
            num_embeddings_per_step,
            embedding_dim,
            commitment_cost,
            decay,
        )
        self.decoder = MAEDecoder(
            image_size, patch_size, embedding_dim, decoder_layer, decoder_head
        )

        self.experience_step = 0
        self.cycle_consistency_power = cycle_consistency_power
        self.cycle_consistency_weight = cycle_consistency_weight

        self.use_lpips = use_lpips
        self._data_variance = data_variance

        if self.use_lpips:
            self._lpips = lpips.LPIPS(net="vgg")

        self.triplet_loss = TripletMarginLoss()

        self.clf_head = nn.Linear(embedding_dim, 2, bias=False)
        self.register_buffer(
            "old_clf_head", torch.zeros((0, embedding_dim), requires_grad=False)
        )

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

    def extend_clf_head(self):
        self.old_clf_head = torch.cat(
            [self.old_clf_head, self.clf_head.weight.data.clone()]
        )
        self.old_clf_head.requires_grad = False

        self.clf_head.weight.data.normal_()

    def calculate_class_maps(self, current_classes: torch.Tensor):
        self.classes_map = {cls_id: i for i, cls_id in enumerate(current_classes)}

    def remap_y(self, y):
        for cls_id, new_cls_id in self.classes_map.items():
            y[y == cls_id] = new_cls_id

        return y

    def get_cycle_consistency_loss(self, distances, indices):
        q_logits = -1 / 2 * distances / self._cycle_consistency_sigma

        q_logits = q_logits.flatten(0, 1)
        q_indices = indices.flatten()

        # Remove loss for mask token
        q_logits = q_logits[q_indices != self._mask_token_id]
        q_indices = q_indices[q_indices != self._mask_token_id]

        return F.cross_entropy(q_logits, q_indices)

    def criterion(self, forward_output: ForwardOutput, y) -> CriterionOutput:
        # prepare default values
        clf_loss = clf_acc = torch.tensor(0.0, device=self.device)
        past_cycle_consistency_loss = torch.tensor(0.0, device=self.device)
        current_cycle_consistency_loss = torch.tensor(0.0, device=self.device)
        reconstruction_loss = torch.tensor(0.0, device=self.device)

        # unpack variables from forward output
        x_recon = forward_output.x_recon
        x_data = forward_output.x_data
        x_indices = forward_output.x_indices

        latent_distances = forward_output.latent_distances
        second_order_distances = forward_output.second_order_distances

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
            current_logits = forward_output.clf_logits[current_data]
            current_y = y[current_data]

            clf_loss = F.cross_entropy(current_logits, current_y)
            clf_acc = (current_logits.argmax(dim=-1) == current_y).float().mean()

        # Compute consistency loss
        if (
            latent_distances is not None
            and self.cycle_consistency_weight != 0
            and past_data.any()
        ):
            distances = latent_distances[past_data]
            indices = x_indices[past_data].long()

            past_cycle_consistency_loss = self.get_cycle_consistency_loss(
                distances, indices
            )

        if (
            second_order_distances is not None
            and self.cycle_consistency_weight != 0
            and current_data.any()
        ):
            distances = second_order_distances[current_data]
            indices = x_indices[current_data].long()

            current_cycle_consistency_loss = self.get_cycle_consistency_loss(
                distances, indices
            )

        # Compute triplet loss
        triplet_loss = self.triplet_loss(
            forward_output.image_emb, forward_output.past_data_mask
        )

        return CriterionOutput(
            vq_loss=forward_output.vq_loss,
            reconstruction_loss=reconstruction_loss,
            past_cycle_consistency_loss=past_cycle_consistency_loss,
            current_cycle_consistency_loss=current_cycle_consistency_loss,
            triplet_loss=triplet_loss,
            clf_loss=clf_loss,
            clf_acc=clf_acc,
            perplexity=forward_output.perplexity,
        )

    def forward(self, x) -> ForwardOutput:
        # prepare default values
        second_order_distances = None

        # Extract features from backbone
        masked_features, full_features, backward_indexes = self.encoder(
            x, return_full_features=True
        )

        with torch.autocast(self._accelerator, dtype=torch.float32):
            (
                vq_loss,
                masked_features,
                perplexity,
                _,
                avg_probs,
                *_,
            ) = self.feature_quantization(masked_features)
            (
                *_,
                x_indices,
                _,
                latent_distances,
            ) = self.feature_quantization(full_features, return_distances=True)

        x_indices = rearrange(x_indices, "(b t) 1 -> b t", b=x.shape[0])
        x_recon, mask = self.decoder(masked_features, backward_indexes)

        # To compute cycle consistency loss we apply encoder/quant again
        if self.cycle_consistency_weight != 0:
            _, second_order_features, _ = self.encoder(
                x_recon, return_full_features=True
            )
            with torch.autocast(self._accelerator, dtype=torch.float32):
                (*_, second_order_distances) = self.feature_quantization(
                    second_order_features, return_distances=True
                )

        # If the model has classification head
        # we calculate image embedding based on output of the encoder
        # without masking random patches
        image_emb = full_features.mean(dim=0)
        clf_logits = torch.cat(
            [self.clf_head(image_emb), image_emb @ self.old_clf_head.T], dim=1
        )

        return ForwardOutput(
            vq_loss=vq_loss,
            x_recon=x_recon,
            x_data=x,
            x_indices=x_indices,
            quantized=masked_features,
            perplexity=perplexity,
            image_emb=image_emb,
            clf_logits=clf_logits,
            mask=mask,
            latent_distances=latent_distances,
            second_order_distances=second_order_distances,
            avg_probs=avg_probs,
            past_data_mask=None,
        )

    def training_step(self, batch, batch_idx):
        data, y, *_ = batch

        x = data["images"]
        past_data = y == -1
        y = self.remap_y(y)

        forward_output = self.forward(x)
        forward_output.x_data = x
        forward_output.past_data_mask = data["time_index"] < self.experience_step
        if past_data.any():
            forward_output.x_indices[past_data] = data["indices"][past_data]

        criterion_output = self.criterion(forward_output, y)

        loss = (
            criterion_output.vq_loss
            + criterion_output.reconstruction_loss
            + criterion_output.past_cycle_consistency_loss
            * self.cycle_consistency_weight
            + criterion_output.current_cycle_consistency_loss
            * self.cycle_consistency_weight
            + criterion_output.triplet_loss
            + criterion_output.clf_loss
        )

        # LOGGING
        self.log_with_postfix(
            f"train/loss",
            loss.cpu().item(),
        )
        self.log_with_postfix(
            f"train/clf_loss",
            criterion_output.clf_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"train/clf_accuracy",
            criterion_output.clf_acc.cpu().item(),
        )
        self.log_with_postfix(
            f"train/triplet_loss",
            criterion_output.triplet_loss.cpu().item(),
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
            f"train/past_cycle_consistency_loss",
            criterion_output.past_cycle_consistency_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"train/current_cycle_consistency_loss",
            criterion_output.current_cycle_consistency_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"train/cycle_consistency_loss",
            (
                criterion_output.past_cycle_consistency_loss.cpu().item()
                + criterion_output.current_cycle_consistency_loss.cpu().item()
            ),
        )

        return {
            "loss": loss,
            "forward_output": forward_output,
        }

    def validation_step(self, batch, batch_idx):
        data, y, *_ = batch

        x = data["images"]
        past_data = y == -1
        y = self.remap_y(y)

        forward_output = self.forward(x)
        forward_output.x_data = x
        forward_output.past_data_mask = data["time_index"] < self.experience_step
        if past_data.any():
            forward_output.x_indices[past_data] = data["indices"][past_data]

        criterion_output = self.criterion(forward_output, y)

        loss = (
            criterion_output.vq_loss
            + criterion_output.reconstruction_loss
            + criterion_output.past_cycle_consistency_loss
            * self.cycle_consistency_weight
            + criterion_output.current_cycle_consistency_loss
            * self.cycle_consistency_weight
            + criterion_output.triplet_loss
            + criterion_output.clf_loss
        )

        # LOGGING
        self.log_with_postfix(
            f"val/loss",
            loss.cpu().item(),
        )
        self.log_with_postfix(
            f"val/clf_loss",
            criterion_output.clf_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"val/clf_accuracy",
            criterion_output.clf_acc.cpu().item(),
        )
        self.log_with_postfix(
            f"val/triplet_loss",
            criterion_output.triplet_loss.cpu().item(),
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
            f"val/past_cycle_consistency_loss",
            criterion_output.past_cycle_consistency_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"val/current_cycle_consistency_loss",
            criterion_output.current_cycle_consistency_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"val/cycle_consistency_loss",
            (
                criterion_output.past_cycle_consistency_loss.cpu().item()
                + criterion_output.current_cycle_consistency_loss.cpu().item()
            ),
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
                self.clf_head.parameters(),
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
