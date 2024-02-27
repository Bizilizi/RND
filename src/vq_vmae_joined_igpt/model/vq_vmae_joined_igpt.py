import dataclasses

import math
import typing as t
from itertools import chain
import lpips
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from transformers import ImageGPTConfig
from src.vq_vmae_joined_igpt.model.image_gpt import ImageGPTForCausalImageModeling

from src.avalanche.model.cl_model import CLModel
from src.vq_vmae_joined_igpt.model.decoder import MAEDecoder
from src.vq_vmae_joined_igpt.model.encoder import MAEEncoder
from src.vq_vmae_joined_igpt.model.quiantizer import (
    VectorQuantizerEMA,
)

if t.TYPE_CHECKING:
    from src.vq_vmae_joined_igpt.model.classification_head import EmbClassifier


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

    # igpt
    igpt_output: torch.Tensor
    input_ids: torch.Tensor


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
    igpt_loss: torch.Tensor


class VQVMAEJoinedIgpt(CLModel):
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
        self.embedding_dim = embedding_dim
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

        # igpt
        self.num_all_embeddings = num_class_embeddings + num_embeddings
        if self.supervised:
            self.sos_token = self.num_all_embeddings + 1 + num_classes
            self.mask_token = self.num_all_embeddings

            vocab_size = self.num_all_embeddings + 1 + num_classes + 1
            n_positions = (16 * 16 + 1) * quantize_top_k + 2
        else:
            self.sos_token = self.num_all_embeddings + 1
            self.mask_token = self.num_all_embeddings

            vocab_size = self.num_all_embeddings + 2
            n_positions = (16 * 16 + 1) * quantize_top_k + 1

        self.num_tokens_without_sos = self.num_all_embeddings + 1
        configuration = ImageGPTConfig(
            **{
                "activation_function": "quick_gelu",
                "attn_pdrop": 0.1,
                "embd_pdrop": 0.1,
                "initializer_range": 0.02,
                "layer_norm_epsilon": 1e-05,
                "model_type": "imagegpt",
                "n_embd": embedding_dim,
                "n_head": 8,
                "n_layer": 12,
                "n_positions": n_positions,
                "reorder_and_upcast_attn": False,
                "resid_pdrop": 0.1,
                "scale_attn_by_inverse_layer_idx": False,
                "scale_attn_weights": True,
                "tie_word_embeddings": False,
                "use_cache": False,
                "vocab_size": vocab_size,
            }
        )
        self.image_gpt = ImageGPTForCausalImageModeling(configuration)

        self.feature_quantization = VectorQuantizerEMA(
            self.num_all_embeddings,
            embedding_dim,
            commitment_cost,
            decay,
            embedding=self.image_gpt.transformer.wte,
            top_k=quantize_top_k,
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

        perplexity = forward_output.feature_perplexity
        cycle_consistency_loss = (
            past_cycle_consistency_loss + current_cycle_consistency_loss
        )

        # igpt
        igpt_output = forward_output.igpt_output
        igpt_loss = F.cross_entropy(
            igpt_output.logits[:, :-1].reshape(-1, igpt_output.logits.shape[-1]),
            forward_output.input_ids[..., 1:].reshape(-1),
        )

        return CriterionOutput(
            vq_loss=forward_output.vq_loss,
            reconstruction_loss=reconstruction_loss,
            past_cycle_consistency_loss=past_cycle_consistency_loss,
            current_cycle_consistency_loss=current_cycle_consistency_loss,
            cycle_consistency_loss=cycle_consistency_loss * 2,
            clf_loss=clf_loss * 1.5,
            clf_acc=clf_acc,
            feature_perplexity=forward_output.feature_perplexity,
            class_perplexity=forward_output.class_perplexity,
            perplexity=perplexity,
            igpt_loss=igpt_loss / 2,
        )

    def forward(self, x, y=None) -> ForwardOutput:
        # prepare default values
        second_order_latent_distances = None
        clf_logits = None
        image_emb = None

        # Extract features from backbone
        masked_features, full_features, backward_indexes = self.encoder(
            x, return_full_features=True
        )

        # Quantize features
        with torch.autocast(self._accelerator, dtype=torch.float32):
            (
                vq_loss,
                masked_features,
                perplexity,
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
        self.decoder.mask_token = self.image_gpt.transformer.wte.weight[self.mask_token]
        x_recon, mask = self.decoder(masked_features, backward_indexes)

        # To compute cycle consistency loss we apply encoder/quant again
        if self._current_cycle_consistency_weight != 0:
            _, second_order_features, _ = self.encoder(
                x_recon, return_full_features=True
            )
            with torch.autocast(self._accelerator, dtype=torch.float32):
                (*_, second_order_latent_distances,) = self.feature_quantization(
                    second_order_features, return_distances=True
                )

        # If the model has classification head we calculate image embedding
        # based on output of the encoder without masking random patches
        if self.clf_head is not None:
            image_emb = full_features.mean(dim=0)
            clf_logits = self.clf_head(image_emb)

        # image gpt
        input_ids = self._rand_mask_indices(z_indices)

        if self.supervised and y is not None:
            input_ids = self._extend_with_classes(y, input_ids)

        input_ids = self._extend_with_sos_token(input_ids)
        igpt_output = self.image_gpt(input_ids=input_ids)

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
            feature_perplexity=perplexity,
            class_perplexity=perplexity,
            # classification
            image_emb=image_emb,
            clf_logits=clf_logits,
            igpt_output=igpt_output,
            input_ids=input_ids,
        )

    def _extend_with_classes(self, y, input_ids):
        classes_ids = y + self.num_tokens_without_sos
        """
        Shift classes ids with (num_embeddings + class_token + mask_token)
        to get classes ids.
        """

        classes_ids = repeat(
            classes_ids,
            "b -> b 1",
        ).to(self.device)

        input_ids = torch.cat([classes_ids, input_ids], dim=1)

        return input_ids

    def _extend_with_sos_token(self, input_ids):
        sos_tokens = torch.full(
            (input_ids.shape[0], 1),
            self.sos_token,
            device=self.device,
        )

        input_ids = torch.cat([sos_tokens, input_ids], dim=1)

        return input_ids

    def _rand_mask_indices(self, indices):
        T = indices.shape[0]
        change_T = int(T * 0.1)

        forward_indices = torch.randperm(T)
        masked_indices = indices.clone()
        masked_indices[forward_indices[:change_T]] = self.mask_token

        return masked_indices

    def training_step(self, batch, batch_idx):
        data, targets, *_ = batch

        x = data["images"]
        y = targets["class"]

        # if self.use_mixup:
        #     x, y_cutmix_or_mixup = self.cutmix_or_mixup(x, y)
        #     targets["y_cutmix_or_mixup"] = y_cutmix_or_mixup

        past_data = targets["time_tag"] == -1

        forward_output = self.forward(x, y)

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
            + criterion_output.igpt_loss
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
            f"train/igpt_loss",
            criterion_output.igpt_loss.cpu().item(),
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
            f"val/igpt_loss",
            criterion_output.igpt_loss.cpu().item(),
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
            self.feature_quantization.parameters(),
            self.decoder.parameters(),
            self.image_gpt.parameters(),
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
