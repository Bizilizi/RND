import dataclasses

import math
import typing as t
from itertools import chain

import lpips
import torch
from einops import rearrange
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ContrastiveLoss, TripletMarginLoss
from timm.models.vision_transformer import Block
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn import functional as F

from src.avalanche.model.cl_model import CLModel
from src.encode_decode_encode.model.decoder import MAEDecoder
from src.encode_decode_encode.model.encoder import MAEEncoder
from src.encode_decode_encode.model.quiantizer import (
    VectorQuantizerEMA,
)


@dataclasses.dataclass
class ForwardOutput:
    vq_loss: torch.Tensor

    x_target: torch.Tensor
    x_recon: torch.Tensor

    z_indices: torch.Tensor
    z_distances: torch.Tensor
    z_indices_target: torch.Tensor

    past_z_indices: torch.Tensor
    past_z_distances: torch.Tensor
    past_z_indices_target: torch.Tensor

    perplexity: torch.Tensor
    avg_probs: torch.Tensor

    clf_logits: torch.Tensor
    clf_targets: torch.Tensor


@dataclasses.dataclass
class CriterionOutput:
    vq_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    latent_consistency_loss: torch.Tensor

    clf_loss: torch.Tensor
    clf_acc: torch.Tensor


class VitVQVae(CLModel):
    def __init__(
        self,
        num_embeddings,
        num_embeddings_per_step,
        embedding_dim,
        img_embedding_dim,
        commitment_cost,
        num_epochs: int,
        batch_size: int,
        num_classes_per_task: int,
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
        reconstruction_loss_weight=1,
        classification_loss_weight=1,
        latent_consistency_loss_weight=1,
        cycle_consistency_sigma: float = 1,
        precision: str = "32-true",
        accelerator: str = "cuda",
        data_variance: float = 1,
    ) -> None:
        super().__init__()

        self.experience_step = 0

        self._num_embeddings = num_embeddings
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._num_epochs = num_epochs
        self._batch_size = batch_size

        self._embedding_dim = embedding_dim
        self._img_embedding_dim = img_embedding_dim

        self._latent_sos_token = num_embeddings + 1
        self._mask_ratio = mask_ratio

        self._precision_dtype = torch.half if precision == "16-mixed" else torch.float32
        self._accelerator = accelerator

        self._cycle_consistency_sigma = cycle_consistency_sigma

        self._data_variance = data_variance

        # Loss weights
        self.lcl_weight = latent_consistency_loss_weight
        self.rec_loss_weight = reconstruction_loss_weight
        self.clf_loss_weight = classification_loss_weight

        # Model layers
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

        self._lpips = lpips.LPIPS(net="vgg")
        for param in self._lpips.parameters():
            param.requires_grad = False

        self.projection_attn = Block(embedding_dim, 4)
        self.projection_head = nn.Linear(embedding_dim, img_embedding_dim)
        self.clf_head = nn.Linear(img_embedding_dim, 10, bias=False)

        self.register_buffer(
            "old_clf_head", torch.zeros((0, img_embedding_dim), requires_grad=False)
        )
        self.register_buffer("old_classes", torch.zeros((0), requires_grad=False))

    def get_reconstruction_loss(self, x: torch.Tensor, x_rec: torch.Tensor):
        lpips_loss = self._lpips(x, x_rec).mean()
        l1_loss = torch.mean(
            F.l1_loss(x, x_rec, reduction="none").mean((1, 2, 3)) / self._data_variance
        )
        reconstruction_loss = lpips_loss + l1_loss

        return reconstruction_loss

    def extend_clf_head(self):
        ...
        # self.old_clf_head = torch.cat([self.old_clf_head, self.clf_head.data.clone()])
        # self.old_clf_head.requires_grad = False
        #
        # self.clf_head.data.normal_()

    def get_cycle_consistency_loss(self, distances, indices):
        q_logits = -1 / 2 * distances / self._cycle_consistency_sigma

        q_logits = q_logits.flatten(0, 1)
        q_indices = indices.flatten()

        # Remove loss for mask token
        mask_token_id = self.feature_quantization.num_embeddings
        q_logits = q_logits[q_indices != mask_token_id]
        q_indices = q_indices[q_indices != mask_token_id]

        return F.cross_entropy(q_logits, q_indices)

    def criterion(self, forward_output: ForwardOutput) -> CriterionOutput:
        # Compute reconstruction loss
        reconstruction_loss = self.get_reconstruction_loss(
            forward_output.x_recon, forward_output.x_target
        )

        # Compute clf loss and accuracy
        clf_loss = F.cross_entropy(
            forward_output.clf_logits, forward_output.clf_targets
        )
        clf_acc = (
            (forward_output.clf_logits.argmax(dim=-1) == forward_output.clf_targets)
            .float()
            .mean()
        )

        # Compute consistency loss
        latent_consistency_loss = torch.tensor(0.0, device=self.device)
        # latent_consistency_loss = self.get_cycle_consistency_loss(
        #     forward_output.z_distances, forward_output.z_indices
        # )

        if forward_output.past_z_indices is not None:
            latent_consistency_loss += self.get_cycle_consistency_loss(
                forward_output.past_z_distances, forward_output.past_z_indices
            )

        return CriterionOutput(
            vq_loss=forward_output.vq_loss,
            reconstruction_loss=reconstruction_loss * self.rec_loss_weight,
            latent_consistency_loss=latent_consistency_loss * self.lcl_weight,
            clf_loss=clf_loss * self.clf_loss_weight,
            clf_acc=clf_acc,
        )

    def get_image_embedding(self, full_features):
        image_emb = self.projection_attn(full_features)[0]
        """ B x emb_dim"""
        image_emb = self.projection_head(image_emb)

        return image_emb

    def present_data_forward(self, x):
        # Performs encode -> decode -> encode

        """
        Notation:
        fo_* - first order
        so_* - second_order
        """
        fo_masked_features, fo_backward_indexes = self.encoder(x)

        # Quantize features
        with torch.autocast(self._accelerator, dtype=torch.float32):
            (
                fo_vq_loss,
                fo_quantized_masked_features,
                fo_perplexity,
                fo_z_indices,
                fo_avg_probs,
                fo_z_distances,
            ) = self.feature_quantization(fo_masked_features, return_distances=True)

        # Compute clf logits
        fo_image_emb = self.get_image_embedding(fo_quantized_masked_features)
        fo_clf_logits = self.clf_head(fo_image_emb)

        # Reconstruct image from masked input
        fo_z_indices = rearrange(fo_z_indices, "(b t) 1 -> b t", b=x.shape[0])
        x_recon, mask = self.decoder(fo_quantized_masked_features, fo_backward_indexes)

        # Encode reconstructed image again to calculate lcl
        so_masked_features, _ = self.encoder(x_recon)

        # Quantize features
        with torch.autocast(self._accelerator, dtype=torch.float32):
            (
                so_vq_loss,
                so_quantized_masked_features,
                so_perplexity,
                so_z_indices,
                so_avg_probs,
                so_z_distances,
            ) = self.feature_quantization(so_masked_features, return_distances=True)

        # Compute clf logits
        so_image_emb = self.get_image_embedding(so_quantized_masked_features)
        so_clf_logits = self.clf_head(so_image_emb)

        so_z_indices = rearrange(
            so_z_indices, "(b t) 1 -> b t", b=fo_z_indices.shape[0]
        )

        return dict(
            mask=mask,
            fo_masked_features=fo_masked_features,
            fo_quantized_masked_features=fo_quantized_masked_features,
            fo_z_indices=fo_z_indices,
            fo_z_distances=fo_z_distances,
            fo_image_emb=fo_image_emb,
            fo_clf_logits=fo_clf_logits,
            x_recon=x_recon,
            fo_vq_loss=fo_vq_loss,
            fo_perplexity=fo_perplexity,
            fo_avg_probs=fo_avg_probs,
            so_masked_features=so_masked_features,
            so_quantized_masked_features=so_quantized_masked_features,
            so_z_indices=so_z_indices,
            so_z_distances=so_z_distances,
            so_image_emb=so_image_emb,
            so_clf_logits=so_clf_logits,
            so_vq_loss=so_vq_loss,
            so_perplexity=so_perplexity,
            so_avg_probs=so_avg_probs,
        )

    def past_data_forward(self, fo_z_indices):
        # Performs decode -> encode -> decode
        # TODO: Make fo_z_indices with masked tokens, ideally randomly populate
        # it as we do it in present_data_forward

        # First we decode image based on its z_indices
        quantized = rearrange(
            self.feature_quantization._embedding(fo_z_indices), "b t c -> t b c"
        )
        features = quantized + self.decoder.pos_embedding

        features = rearrange(features, "t b c -> b t c")
        features = self.decoder.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # remove global feature

        patches = self.decoder.head(features)
        x_recon = self.decoder.patch2img(patches)

        # Now we encode and decode again
        """
        so_* - second order
        """

        so_masked_features, so_backward_indexes = self.encoder(x_recon, ratio=0)

        # Quantize features
        with torch.autocast(self._accelerator, dtype=torch.float32):
            (
                so_vq_loss,
                so_quantized_masked_features,
                so_perplexity,
                so_z_indices,
                so_avg_probs,
                so_z_distances,
            ) = self.feature_quantization(so_masked_features, return_distances=True)

        # Compute clf logits
        so_image_emb = self.get_image_embedding(so_quantized_masked_features)
        so_clf_logits = self.clf_head(so_image_emb)

        # Reconstruct image from masked input
        so_z_indices = rearrange(
            so_z_indices, "(b t) 1 -> b t", b=fo_z_indices.shape[0]
        )
        so_x_recon, _ = self.decoder(so_quantized_masked_features, so_backward_indexes)

        return dict(
            x_recon=x_recon,
            so_masked_features=so_masked_features,
            so_quantized_masked_features=so_quantized_masked_features,
            so_z_indices=so_z_indices,
            so_z_distances=so_z_distances,
            so_image_emb=so_image_emb,
            so_clf_logits=so_clf_logits,
            so_vq_loss=so_vq_loss,
            so_perplexity=so_perplexity,
            so_avg_probs=so_avg_probs,
            so_x_recon=so_x_recon,
        )

    def forward(self, data, y) -> ForwardOutput:
        # default past variables
        past_z_indices = None
        past_z_distances = None
        past_z_indices_target = None

        past_data_mask = data["is_past_domain"] == 1
        present_data_mask = ~past_data_mask

        present_input = data["images"][present_data_mask]
        present_forward_output = self.present_data_forward(present_input)

        # create two image tensors to compare
        x_recon = present_forward_output["x_recon"]
        x_target = present_input

        # create z_indices tensors to compare
        z_indices = present_forward_output["so_z_indices"]
        z_distances = present_forward_output["so_z_distances"]
        z_indices_target = present_forward_output["fo_z_indices"].detach()

        # create clf logits and targets
        clf_logits = torch.cat(
            [
                present_forward_output["fo_clf_logits"],
                present_forward_output["so_clf_logits"],
            ]
        )
        clf_targets = torch.cat([y[present_data_mask], y[present_data_mask]])

        # crate VQ-VAE losses variables
        vq_loss = (
            present_forward_output["fo_vq_loss"] + present_forward_output["so_vq_loss"]
        )
        perplexity = (
            present_forward_output["fo_perplexity"]
            + present_forward_output["so_perplexity"]
        )
        avg_probs = (
            present_forward_output["fo_avg_probs"]
            + present_forward_output["so_avg_probs"]
        ) / 2

        if past_data_mask.any():
            past_input = data["indices"][past_data_mask]
            past_forward_output = self.past_data_forward(past_input)

            # extend reconstruction objectives
            x_recon = torch.cat([x_recon, past_forward_output['so_x_recon']])
            x_target = torch.cat(
                [x_target, past_forward_output['x_recon'].clone().detach()]
            )

            # extend lcl objective (lcl - latent consistency loss)
            past_z_indices = past_forward_output["so_z_indices"]
            past_z_distances = past_forward_output["so_z_distances"]
            past_z_indices_target = past_input

            # extend clf objective
            clf_logits = torch.cat([clf_logits, past_forward_output["so_clf_logits"]])
            clf_targets = torch.cat([clf_targets, y[past_data_mask]])

            # extend vq-vae objectives
            vq_loss = vq_loss + past_forward_output["so_vq_loss"]
            perplexity = perplexity + past_forward_output["so_perplexity"]
            avg_probs = (avg_probs * 2 + past_forward_output["so_avg_probs"]) / 3

        return ForwardOutput(
            x_recon=x_recon,
            x_target=x_target,
            z_indices=z_indices,
            z_distances=z_distances,
            z_indices_target=z_indices_target,
            past_z_indices=past_z_indices,
            past_z_distances=past_z_distances,
            past_z_indices_target=past_z_indices_target,
            clf_logits=clf_logits,
            clf_targets=clf_targets,
            vq_loss=vq_loss,
            perplexity=perplexity,
            avg_probs=avg_probs,
        )

    def training_step(self, batch, batch_idx):
        data, y, *_ = batch

        forward_output = self.forward(data, y)
        criterion_output = self.criterion(forward_output)

        loss = (
            criterion_output.vq_loss
            + criterion_output.reconstruction_loss
            + criterion_output.latent_consistency_loss
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
            f"train/vq_loss",
            criterion_output.vq_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"train/reconstruction_loss",
            criterion_output.reconstruction_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"train/perplexity",
            forward_output.perplexity.cpu().item(),
        )

        self.log_with_postfix(
            f"train/latent_consistency_loss",
            criterion_output.latent_consistency_loss.cpu().item(),
        )

        return {
            "loss": loss,
            "forward_output": forward_output,
        }

    def validation_step(self, batch, batch_idx):
        data, y, *_ = batch

        forward_output = self.forward(data, y)
        criterion_output = self.criterion(forward_output)

        loss = (
            criterion_output.vq_loss
            + criterion_output.reconstruction_loss
            + criterion_output.latent_consistency_loss
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
            f"val/vq_loss",
            criterion_output.vq_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"val/reconstruction_loss",
            criterion_output.reconstruction_loss.cpu().item(),
        )
        self.log_with_postfix(
            f"val/perplexity",
            forward_output.perplexity.cpu().item(),
        )
        self.log_with_postfix(
            f"val/latent_consistency_loss",
            criterion_output.latent_consistency_loss.cpu().item(),
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
                self.projection_attn.parameters(),
                self.projection_head.parameters(),
                self.clf_head.parameters(),
            ),
            lr=self._learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self._weight_decay,
        )

        return optimizer

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
