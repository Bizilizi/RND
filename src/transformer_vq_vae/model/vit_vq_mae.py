import dataclasses
import math
import typing as t
from itertools import chain

import lpips
import torch
from einops import rearrange
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F

import wandb
from avalanche.benchmarks import CLExperience, NCExperience
from src.avalanche.model.cl_model import CLModel
from src.transformer_vq_vae.model.decoder import MAEDecoder
from src.transformer_vq_vae.model.encoder import MAEEncoder
from src.transformer_vq_vae.model.quiantizer import (
    FeatureQuantizerEMA,
    SeparateCodebooksFeatureQuantizerEMA,
)

if t.TYPE_CHECKING:
    from src.transformer_vq_vae.model.classification_head import EmbClassifier


@dataclasses.dataclass
class ForwardOutput:
    """"""

    """
    Tensors with past and current images. (original and reconstruction)
    """
    x_data: torch.Tensor
    x_recon: torch.Tensor

    """
    Mask used to cover image patches
    """
    mask: torch.Tensor

    """
    First order features.
    x -(enc)-> z 
    """
    z_quantized: torch.Tensor
    z_distances: torch.Tensor
    z_indices: torch.Tensor

    """
    Second order features
    x -(enc)-> z -(dec)-> x' -(enc)-> z'
    """
    z_second_order_distances: torch.Tensor

    """
    Metrics calculated on forward pass
    """
    vq_loss: torch.Tensor
    perplexity: torch.Tensor
    avg_probs: torch.Tensor

    # classification
    image_emb: torch.Tensor
    clf_logits: torch.Tensor


@dataclasses.dataclass
class CriterionOutput:
    reconstruction_loss: torch.Tensor
    past_reconstruction_loss: torch.Tensor
    current_reconstruction_loss: torch.Tensor
    past_cycle_consistency_loss: torch.Tensor
    current_cycle_consistency_loss: torch.Tensor
    cycle_consistency_loss: torch.Tensor

    # classification loss
    clf_loss: torch.Tensor
    clf_acc: torch.Tensor

    # quantization loss
    vq_loss: torch.Tensor
    perplexity: torch.Tensor


class VQMAE(CLModel):
    def __init__(
        self,
        *,
        # Mmodel parameters
        num_embeddings,
        num_embeddings_per_step,
        embedding_dim,
        decoder_embedding_dim,
        commitment_cost,
        mask_token_id: int,
        decay=0,
        weight_decay=0.05,
        image_size=32,
        patch_size=2,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
        mask_ratio=0.75,
        use_lpips: bool = True,
        # training coefficients
        num_epochs: int,
        batch_size: int,
        learning_rate: float = 1e-3,
        warmup: float = 200,
        # supervision
        supervised: bool = False,
        num_classes: int = 0,
        # loss coefficients
        cycle_consistency_sigma: float = 1,
        past_cycle_consistency_weight=1,
        current_cycle_consistency_weight=1,
        past_samples_loss_weight=1,
        current_samples_loss_weight=1,
        future_samples_loss_weight=1,
        data_variance=0.06328692405746414,
        # quantization params
        quantize_features: bool = True,
        quantize_top_k: int = 3,
        perplexity_threshold: float = 0,
    ) -> None:
        super().__init__()

        # Model parameters
        self._num_embeddings = num_embeddings
        self._weight_decay = weight_decay
        self._embedding_dim = embedding_dim
        self._latent_sos_token = num_embeddings + 1
        self._mask_ratio = mask_ratio
        self._mask_token_id = mask_token_id
        self._supervised = supervised
        self._use_lpips = use_lpips

        # Training coefficients
        self._warmup = warmup
        self._learning_rate = learning_rate
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self.experience_step = 0

        # Loss coefficients
        self._data_variance = data_variance
        self._past_samples_loss_weight = past_samples_loss_weight
        self._current_samples_loss_weight = current_samples_loss_weight
        self._future_samples_loss_weight = future_samples_loss_weight

        self._cycle_consistency_sigma = cycle_consistency_sigma
        self._current_cycle_consistency_weight = current_cycle_consistency_weight
        self._past_cycle_consistency_weight = past_cycle_consistency_weight

        # Quantization flags
        self._quantize_features = quantize_features
        self._quantize_top_k = quantize_top_k

        self.encoder = MAEEncoder(
            image_size,
            patch_size,
            embedding_dim,
            encoder_layer,
            encoder_head,
        )

        self.decoder = MAEDecoder(
            image_size, patch_size, decoder_embedding_dim, decoder_layer, decoder_head
        )

        self.feature_quantization = FeatureQuantizerEMA(
            num_embeddings,
            num_embeddings_per_step,
            embedding_dim,
            commitment_cost,
            decay,
            top_k=quantize_top_k,
            perplexity_threshold=perplexity_threshold,
        )

        if self._supervised:
            self.clf_head = nn.Linear(embedding_dim, num_classes)

        else:
            self.clf_head = None

        if self._use_lpips:
            self._lpips = lpips.LPIPS(net="vgg")

    def set_clf_head(self, model: "EmbClassifier"):
        self.__dict__["clf_head"] = model

    def reset_clf_head(self):
        self.clf_head = None

    def update_model_experience(
        self, experience_step: int, experience: t.Union[CLExperience, NCExperience]
    ) -> None:
        self.experience_step = experience_step
        self.experience = experience

        self.feature_quantization.update_model_experience(experience_step)

    def get_experience_step(self):
        return self.experience_step

    def get_experience(self):
        return self.experience

    def set_experience_step(self, experience_step):
        self.experience_step = experience_step

    def set_experience(self, experience):
        self.experience = experience

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        x_rec: torch.Tensor,
        past_data: torch.Tensor,
        current_data: torch.Tensor,
    ):

        # Create weight vector to shift gradient towards current dataset
        weight_tensor = torch.ones(past_data.shape[0], device=self.device)
        weight_tensor[past_data] = self._past_samples_loss_weight
        weight_tensor[current_data] = self._current_samples_loss_weight

        past_l1_reconstruction_loss = (
            F.l1_loss(x, x_rec, reduction="none").mean((1, 2, 3)) * past_data
        )

        current_l1_reconstruction_loss = (
            F.l1_loss(x, x_rec, reduction="none").mean((1, 2, 3)) * current_data
        )

        reconstruction_loss = torch.mean(
            (past_l1_reconstruction_loss + current_l1_reconstruction_loss)
            * weight_tensor
            / self._data_variance
        )

        if self._use_lpips:
            lpips_loss = (self._lpips(x, x_rec) * weight_tensor).mean()
            reconstruction_loss += lpips_loss

        return (
            reconstruction_loss,
            past_l1_reconstruction_loss,
            current_l1_reconstruction_loss,
        )

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

    def criterion(self, forward_output: ForwardOutput, y) -> CriterionOutput:
        # prepare default values
        clf_loss = torch.tensor(0.0, device=self.device)
        clf_acc = torch.tensor(0.0, device=self.device)
        past_cycle_consistency_loss = torch.tensor(0.0, device=self.device)
        current_cycle_consistency_loss = torch.tensor(0.0, device=self.device)

        # unpack variables from forward output for readability
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

        past_data = y["time_tag"] == -1
        current_data = y["time_tag"] == 0
        target_class = y["class"]

        # Compute reconstruction loss for past and current data
        (
            reconstruction_loss,
            past_l1_reconstruction_loss,
            current_l1_reconstruction_loss,
        ) = self.get_reconstruction_loss(x_recon, x_data, past_data, current_data)

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
            clf_loss = F.cross_entropy(forward_output.clf_logits, target_class)
            clf_acc = (
                (forward_output.clf_logits.argmax(dim=-1) == target_class)
                .float()
                .mean()
            )

        cycle_consistency_loss = (
            past_cycle_consistency_loss + current_cycle_consistency_loss
        )
        return CriterionOutput(
            vq_loss=forward_output.vq_loss,
            reconstruction_loss=reconstruction_loss,
            past_reconstruction_loss=past_l1_reconstruction_loss,
            current_reconstruction_loss=current_l1_reconstruction_loss,
            past_cycle_consistency_loss=past_cycle_consistency_loss,
            current_cycle_consistency_loss=current_cycle_consistency_loss,
            cycle_consistency_loss=cycle_consistency_loss,
            clf_loss=clf_loss,
            clf_acc=clf_acc,
            perplexity=forward_output.perplexity,
        )

    def forward(self, x) -> ForwardOutput:
        # prepare default values
        z_second_order_distances = None
        clf_logits = None
        image_emb = None

        # Extract features from backbone
        masked_features, full_features, backward_indexes = self.encoder(
            x, return_full_features=True
        )

        # Quantize features
        masked_quantization = self.feature_quantization(masked_features)

        masked_features = masked_quantization.quantized
        """
        masked_features shape - T x B x top_k x emb_dim
        """

        masked_features = masked_features.mean(2)
        """
        masked_features shape - T x B x emb_dim
        """
        full_quantization = self.feature_quantization(full_features)
        z_indices = rearrange(full_quantization.encoding_indices, "t b k -> b (t k)")

        # Reconstruct an image from quantized patches' features
        x_recon, mask = self.decoder(masked_features, backward_indexes)

        # To compute cycle consistency loss we apply encoder/quant again
        if self._current_cycle_consistency_weight != 0:
            _, second_order_features, _ = self.encoder(
                x_recon, return_full_features=True
            )
            if self._quantize_features:
                second_order_quantization = self.feature_quantization(
                    second_order_features
                )
                z_second_order_distances = second_order_quantization.distances

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
            z_distances=full_quantization.distances,
            x_recon=x_recon,
            # seconds order
            z_second_order_distances=z_second_order_distances,
            # metrics
            vq_loss=masked_quantization.loss,
            perplexity=masked_quantization.perplexity,
            avg_probs=masked_quantization.avg_probs,
            # classification
            image_emb=image_emb,
            clf_logits=clf_logits,
        )

    def training_step(self, batch, batch_idx):
        data, y, *_ = batch

        x = data["images"]
        past_data = y["time_tag"] == -1

        forward_output = self.forward(x)

        """ 
        Because x contains both past and present images.
        The forward_output.z_indices will contain indices for latent codes
        computed with respect to the current state of weights in VQ-MAE encoder for 
        past and present images as well.
        
        However it is not true, so we overwrite it with data from the batch, 
        that contains correct indices from previous state of weights in VQ-MAE, by dataset design.
        
        z_indices now contains proper target indices for past and current data
        """
        if past_data.any():
            forward_output.z_indices[past_data] = data["indices"][past_data]

        criterion_output = self.criterion(forward_output, y)

        loss = (
            criterion_output.vq_loss
            + criterion_output.reconstruction_loss
            + criterion_output.cycle_consistency_loss
        )

        if self._supervised:
            loss += criterion_output.clf_loss
            self.log_with_postfix(
                "train/classification_loss",
                criterion_output.clf_loss.cpu().item(),
            )
            self.log_with_postfix(
                "train/classification_accuracy",
                criterion_output.clf_acc.cpu().item(),
            )

        # LOGGING
        self.log_with_postfix(
            "train/loss",
            loss.cpu().item(),
        )
        self.log_with_postfix(
            "train/vq_loss",
            criterion_output.vq_loss.cpu().item(),
        )
        self.log_with_postfix(
            "train/reconstruction_loss",
            criterion_output.reconstruction_loss.cpu().item(),
        )
        self.log_with_postfix(
            "train/reconstruction_l1_loss_past",
            criterion_output.past_reconstruction_loss.cpu().item(),
        )
        self.log_with_postfix(
            "train/reconstruction_l1_loss_current",
            criterion_output.current_reconstruction_loss.cpu().item(),
        )
        self.log_with_postfix(
            "train/perplexity",
            criterion_output.perplexity.cpu().item(),
        )
        self.log_with_postfix(
            "train/current_cycle_consistency_loss",
            criterion_output.current_cycle_consistency_loss.cpu().item(),
        )
        self.log_with_postfix(
            "train/past_cycle_consistency_loss",
            criterion_output.past_cycle_consistency_loss.cpu().item(),
        )
        self.log_with_postfix(
            "train/cycle_consistency_loss",
            criterion_output.cycle_consistency_loss.cpu().item(),
        )

        return {
            "loss": loss,
            "forward_output": forward_output,
        }

    def on_train_epoch_end(self) -> None:

        # schedulers
        sch = self.lr_schedulers()
        sch.step()

    def validation_step(self, batch, batch_idx):
        data, y, *_ = batch

        x = data["images"]
        past_data = y["time_tag"] == -1

        forward_output = self.forward(x)

        if past_data.any():
            forward_output.z_indices[past_data] = data["indices"][past_data]

        criterion_output = self.criterion(forward_output, y)

        loss = (
            criterion_output.vq_loss
            + criterion_output.reconstruction_loss
            + criterion_output.cycle_consistency_loss
        )

        if self._supervised:
            loss += criterion_output.clf_loss
            self.log_with_postfix(
                "val/classification_loss",
                criterion_output.clf_loss.cpu().item(),
            )
            self.log_with_postfix(
                "val/classification_accuracy",
                criterion_output.clf_acc.cpu().item(),
            )

        # LOGGING
        self.log_with_postfix(
            "val/loss",
            loss.cpu().item(),
        )
        self.log_with_postfix(
            "val/vq_loss",
            criterion_output.vq_loss.cpu().item(),
        )
        self.log_with_postfix(
            "val/reconstruction_loss",
            criterion_output.reconstruction_loss.cpu().item(),
        )
        self.log_with_postfix(
            "val/reconstruction_l1_loss_past",
            criterion_output.past_reconstruction_loss.cpu().item(),
        )
        self.log_with_postfix(
            "val/reconstruction_l1_loss_current",
            criterion_output.current_reconstruction_loss.cpu().item(),
        )
        self.log_with_postfix(
            "val/perplexity",
            criterion_output.perplexity.cpu().item(),
        )
        self.log_with_postfix(
            "val/cycle_consistency_loss",
            criterion_output.cycle_consistency_loss.cpu().item(),
        )

        return {
            "loss": loss,
            "forward_output": forward_output,
        }

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

        lr_func = lambda epoch: min(
            (epoch + 1) / (self._warmup + 1e-8),
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
