import dataclasses

import math
import typing as t
from itertools import chain
from torchvision.transforms import v2
import lpips
import torch
from einops import rearrange
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ContrastiveLoss
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from src.avalanche.model.cl_model import CLModel
from src.transformer_vq_vae.model.decoder import MAEDecoder
from src.transformer_vq_vae.model.encoder import MAEEncoder
from src.transformer_vq_vae.model.quiantizer import (
    VectorQuantizerEMA,
    FeatureQuantizer,
)

if t.TYPE_CHECKING:
    from src.transformer_vq_vae.model.classification_head import EmbClassifier


@dataclasses.dataclass
class ForwardOutput:
    # data input
    x_data: torch.Tensor
    mask: torch.Tensor

    # first order features
    z_quantized: torch.Tensor
    z_distances: torch.Tensor
    z_indices: torch.Tensor
    x_recon: torch.Tensor

    # second order features
    z_second_order_distances: torch.Tensor

    # metrics
    vq_loss: torch.Tensor
    feature_perplexity: torch.Tensor
    class_perplexity: torch.Tensor

    # classification
    image_emb: torch.Tensor
    clf_logits: torch.Tensor


@dataclasses.dataclass
class CriterionOutput:
    reconstruction_loss: torch.Tensor
    past_cycle_consistency_loss: torch.Tensor
    current_cycle_consistency_loss: torch.Tensor
    cycle_consistency_loss: torch.Tensor

    # classification loss
    clf_loss: torch.Tensor
    clf_acc: torch.Tensor

    # quantization loss
    vq_loss: torch.Tensor
    feature_perplexity: torch.Tensor
    class_perplexity: torch.Tensor
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
        num_classes: int,
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
        cycle_consistency_sigma: float = 1,
        past_cycle_consistency_weight=1,
        current_cycle_consistency_weight=1,
        past_samples_loss_weight=1,
        current_samples_loss_weight=1,
        future_samples_loss_weight=1,
        precision: str = "32-true",
        accelerator: str = "cuda",
        quantize_features: bool = True,
        quantize_top_k: int = 3,
        separate_codebooks: bool = True,
        supervised: bool = False,
        use_mixup: bool = False,
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
        self.supervised = supervised

        if precision == "16-mixed" and accelerator == "cpu":
            self._precision_dtype = torch.bfloat16
        elif precision == "16-mixed":
            self._precision_dtype = torch.half
        else:
            self._precision_dtype = torch.float32

        self._accelerator = accelerator
        self._num_epochs = num_epochs
        self._batch_size = batch_size

        self._past_samples_loss_weight = past_samples_loss_weight
        self._current_samples_loss_weight = current_samples_loss_weight
        self._future_samples_loss_weight = future_samples_loss_weight

        self._cycle_consistency_sigma = cycle_consistency_sigma
        self._current_cycle_consistency_weight = current_cycle_consistency_weight
        self._past_cycle_consistency_weight = past_cycle_consistency_weight

        self._quantize_features = quantize_features
        self._quantize_top_k = quantize_top_k

        self.encoder = MAEEncoder(
            image_size,
            patch_size,
            embedding_dim,
            encoder_layer,
            encoder_head,
        )
        self.feature_quantization = FeatureQuantizer(
            num_class_embeddings,
            num_embeddings,
            embedding_dim,
            commitment_cost,
            decay,
            top_k=quantize_top_k,
            separate_codebooks=separate_codebooks,
        )
        self.decoder = MAEDecoder(
            image_size, patch_size, embedding_dim, decoder_layer, decoder_head
        )

        if self.supervised:
            self.clf_head = nn.Linear(embedding_dim, num_classes)
        else:
            self.clf_head = None

        self.experience_step = 0
        self.use_lpips = use_lpips
        self._data_variance = 0.06328692405746414

        if self.use_lpips:
            self._lpips = lpips.LPIPS(net="vgg")

        self.use_mixup = use_mixup
        if use_mixup:
            cutmix = v2.CutMix(num_classes=num_classes)
            mixup = v2.MixUp(num_classes=num_classes)
            self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def set_clf_head(self, model: "EmbClassifier"):
        self.__dict__["clf_head"] = model

    def reset_clf_head(self):
        self.clf_head = None

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        x_rec: torch.Tensor,
        past_data: torch.Tensor,
        current_data: torch.Tensor,
    ):

        # Create weight vector to shift gradient towards current dataset
        weight_tensor = torch.ones(x.shape[0], device=self.device)
        weight_tensor[past_data] = self._past_samples_loss_weight
        weight_tensor[current_data] = self._current_samples_loss_weight

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

    def get_cycle_consistency_loss(self, distances, indices):
        indices = rearrange(indices, "b (t k) -> b t k", k=self._quantize_top_k)
        indices = indices[..., 0]

        q_logits = -1 / 2 * distances / self._cycle_consistency_sigma

        q_logits = q_logits.flatten(0, 1)
        q_indices = indices.flatten()

        # Remove loss for mask token if it presents there
        q_logits = q_logits[q_indices != self._mask_token_id]
        q_indices = q_indices[q_indices != self._mask_token_id]

        return F.cross_entropy(q_logits, q_indices)

    def criterion(self, forward_output: ForwardOutput, targets) -> CriterionOutput:
        # prepare default values
        clf_loss = clf_acc = torch.tensor(0.0, device=self.device)
        past_cycle_consistency_loss = torch.tensor(0.0, device=self.device)
        current_cycle_consistency_loss = torch.tensor(0.0, device=self.device)
        reconstruction_loss = torch.tensor(0.0, device=self.device)

        # unpack variables from forward output
        x_recon = forward_output.x_recon
        x_data = forward_output.x_data
        z_indices = forward_output.z_indices

        z_distances = forward_output.z_distances
        if z_distances is not None:
            z_distances = rearrange(z_distances, "t b c -> b t c")

        z_second_order_distances = forward_output.z_second_order_distances
        if z_second_order_distances is not None:
            z_second_order_distances = rearrange(
                z_second_order_distances, "t b c -> b t c"
            )

        y = targets["class"]
        y_cutmix_or_mixup = targets.get("y_cutmix_or_mixup", y)

        past_data = targets["time_tag"] == -1
        current_data = targets["time_tag"] == 0

        # Compute reconstruction loss for future and current data
        reconstruction_loss = self.get_reconstruction_loss(
            x_recon,
            x_data,
            past_data,
            current_data,
        )

        # Compute consistency loss for past data
        if (
            z_distances is not None
            and self._past_cycle_consistency_weight != 0
            and past_data.any()
        ):
            distances = z_distances[past_data]
            indices = z_indices[past_data].long()

            past_cycle_consistency_loss = (
                self.get_cycle_consistency_loss(distances, indices)
                * self._past_cycle_consistency_weight
            )

        # Compute consistency loss current data
        if (
            z_second_order_distances is not None
            and self._current_cycle_consistency_weight != 0
            and current_data.any()
            and self.current_epoch >= self._num_epochs // 2
        ):
            distances = z_second_order_distances[current_data]
            indices = z_indices[current_data].long()

            current_cycle_consistency_loss = (
                self.get_cycle_consistency_loss(distances, indices)
                * self._current_cycle_consistency_weight
            )

        # Compute accuracy if classification head presents
        if forward_output.clf_logits is not None:
            clf_loss = F.cross_entropy(forward_output.clf_logits, y_cutmix_or_mixup)
            clf_acc = (forward_output.clf_logits.argmax(dim=-1) == y).float().mean()

        perplexity = forward_output.feature_perplexity + forward_output.class_perplexity
        cycle_consistency_loss = (
            past_cycle_consistency_loss + current_cycle_consistency_loss
        )
        return CriterionOutput(
            vq_loss=forward_output.vq_loss,
            reconstruction_loss=reconstruction_loss,
            past_cycle_consistency_loss=past_cycle_consistency_loss,
            current_cycle_consistency_loss=current_cycle_consistency_loss,
            cycle_consistency_loss=cycle_consistency_loss,
            clf_loss=clf_loss,
            clf_acc=clf_acc,
            feature_perplexity=forward_output.feature_perplexity,
            class_perplexity=forward_output.class_perplexity,
            perplexity=perplexity,
        )

    def forward(self, x) -> ForwardOutput:
        # prepare default values
        latent_distances = None
        second_order_latent_distances = None
        feature_perplexity = torch.tensor(0.0, device=self.device)
        class_perplexity = torch.tensor(0.0, device=self.device)
        vq_loss = torch.tensor(0.0, device=self.device)
        clf_logits = None
        image_emb = None

        # Extract features from backbone
        masked_features, full_features, backward_indexes = self.encoder(
            x, return_full_features=True
        )

        # Quantize features
        if self._quantize_features:
            with torch.autocast(self._accelerator, dtype=torch.float32):
                (
                    vq_loss,
                    masked_features,
                    feature_perplexity,
                    class_perplexity,
                    *_,
                ) = self.feature_quantization(masked_features)
                """
                masked_features shape - T x B x top_k x emb_dim
                """
                masked_features = masked_features.mean(2)
                """
                masked_features shape - T x B x emb_dim
                """
                (*_, z_indices, latent_distances) = self.feature_quantization(
                    full_features, return_distances=True
                )
                z_indices = rearrange(z_indices, "t b k -> b (t k)")

        # Reconstruct an image from quantized patches' features
        x_recon, mask = self.decoder(masked_features, backward_indexes)

        # To compute cycle consistency loss we apply encoder/quant again
        if self._current_cycle_consistency_weight != 0:
            _, second_order_features, _ = self.encoder(
                x_recon, return_full_features=True
            )
            if self._quantize_features:
                with torch.autocast(self._accelerator, dtype=torch.float32):
                    (*_, second_order_latent_distances) = self.feature_quantization(
                        second_order_features, return_distances=True
                    )

        # If the model has classification head we calculate image embedding
        # based on output of the encoder without masking random patches
        if self.clf_head is not None:
            image_emb = full_features.mean(dim=0)
            clf_logits = self.clf_head(image_emb)

        return ForwardOutput(
            # data
            x_data=x,
            mask=mask,
            # first order
            z_quantized=masked_features,
            z_indices=z_indices,
            z_distances=latent_distances,
            x_recon=x_recon,
            # seconds order
            z_second_order_distances=second_order_latent_distances,
            # metrics
            vq_loss=vq_loss,
            feature_perplexity=feature_perplexity,
            class_perplexity=class_perplexity,
            # classification
            image_emb=image_emb,
            clf_logits=clf_logits,
        )

    def training_step(self, batch, batch_idx):
        data, targets, *_ = batch

        x = data["images"]
        y = targets["class"]

        if self.use_mixup:
            x, y_cutmix_or_mixup = self.cutmix_or_mixup(x, y)
            targets["y_cutmix_or_mixup"] = y_cutmix_or_mixup

        past_data = targets["time_tag"] == -1

        forward_output = self.forward(x)

        # The forward_output.z_indices contains indices for latent codes
        # we overwrite it with data from the batch to make sure
        # that z_indices now contains target indices for past and current data
        if past_data.any():
            forward_output.z_indices[past_data] = data["indices"][past_data]

        criterion_output = self.criterion(forward_output, targets)

        loss = (
            criterion_output.vq_loss
            + criterion_output.reconstruction_loss
            + criterion_output.cycle_consistency_loss
        )

        if self.supervised:
            loss += criterion_output.clf_loss

            self.log_with_postfix(
                f"train/classification_loss",
                criterion_output.clf_loss.cpu().item(),
            )
            self.log_with_postfix(
                f"train/classification_accuracy",
                criterion_output.clf_acc.cpu().item(),
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
            f"train/class_perplexity",
            criterion_output.class_perplexity.cpu().item(),
        )
        self.log_with_postfix(
            f"train/feature_perplexity",
            criterion_output.feature_perplexity.cpu().item(),
        )
        self.log_with_postfix(
            f"train/current_cycle_consistency_loss",
            criterion_output.current_cycle_consistency_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"train/past_cycle_consistency_loss",
            criterion_output.past_cycle_consistency_loss.cpu().item(),
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
        data, targets, *_ = batch

        x = data["images"]
        y = targets["class"]

        past_data = targets["time_tag"] == -1

        forward_output = self.forward(x)

        if past_data.any():
            forward_output.z_indices[past_data] = data["indices"][past_data]

        criterion_output = self.criterion(forward_output, targets)

        loss = (
            criterion_output.vq_loss
            + criterion_output.reconstruction_loss
            + criterion_output.cycle_consistency_loss
        )

        if self.supervised:
            loss += criterion_output.clf_loss

            self.log_with_postfix(
                f"val/classification_loss",
                criterion_output.clf_loss.cpu().item(),
            )
            self.log_with_postfix(
                f"val/classification_accuracy",
                criterion_output.clf_acc.cpu().item(),
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
            f"val/class_perplexity",
            criterion_output.class_perplexity.cpu().item(),
        )
        self.log_with_postfix(
            f"val/feature_perplexity",
            criterion_output.feature_perplexity.cpu().item(),
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
        parameters = [
            self.encoder.parameters(),
            # self.feature_quantization.parameters(),
            self.decoder.parameters(),
        ]

        if self.supervised:
            parameters.append(self.clf_head.parameters())

        optimizer = torch.optim.AdamW(
            chain(*parameters),
            lr=self._learning_rate * self._batch_size / 256,
            betas=(0.9, 0.95),
            weight_decay=self._weight_decay,
        )

        lr_func = lambda epoch: min(
            (epoch + 1) / (200 + 1e-8),
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
            *args,
            **kwargs,
            sync_dist=True,
        )

    def unfreeze(self) -> None:
        super().unfreeze()

        for param in self.feature_quantization.parameters():
            param.requires_grad = False
