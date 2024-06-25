import dataclasses

import typing as t
from itertools import chain

import lpips
import torch
from einops import rearrange
from torch.nn import functional as F

from src.avalanche.model.cl_model import CLModel
from src.qmae_memory.model.loss.patch_gan_discriminator import (
    NLayerDiscriminator,
    weights_init,
)
from src.qmae_memory.model.loss.patch_vit_discriminator import PatchVITDiscriminator
from src.qmae_memory.model.mae.decoder import MAEDecoder
from src.qmae_memory.model.mae.encoder import MAEEncoder
from src.qmae_memory.model.vqvae.quiantizer import (
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

    # masking arguments
    present_forward_indexes: torch.Tensor
    present_remain_T: int
    past_forward_indexes: t.Optional[torch.Tensor]
    past_remain_T: t.Optional[int]


@dataclasses.dataclass
class ReconstrcutionCriterionOutput:
    loss: torch.Tensor
    vq_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    latent_consistency_loss: torch.Tensor
    generator_loss: torch.Tensor


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


class QMAE(CLModel):
    def __init__(
        self,
        *,
        # quantisation
        commitment_cost,
        decay=0,
        num_embeddings,
        num_embeddings_per_step,
        embedding_dim,
        img_embedding_dim,
        # mae
        image_size=32,
        patch_size=2,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
        # discriminator
        discriminator_type='patch-vit',
        gan_loss_epoch_start,
        disc_num_layers=3,
        disc_num_heads=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_use_actnorm=False,
        disc_ndf=64,
        disc_loss="hinge",
        disc_train_steps=10,
        # loss weights
        l1_loss_weight: float = 1,
        lpip_loss_weight: float = 1,
        vq_loss_weight: float = 1,
        latent_consistency_loss_weight: float = 1,
        cycle_consistency_sigma: float = 1,
        discriminator_weight: float = 1,
        # training
        precision: str = "32-true",
        accelerator: str = "cuda",
        learning_rate: float = 1e-3,
        weight_decay=0.05,
        batch_size: int,
        accumulate_batch_every: int,
        num_epochs: int,
    ) -> None:
        super().__init__()

        # turn of automatic optimisation
        self.automatic_optimization = False

        self.experience_step = 0

        self.num_embeddings = num_embeddings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.accumulate_batch_every = accumulate_batch_every

        self.embedding_dim = embedding_dim
        self.img_embedding_dim = img_embedding_dim

        self.latent_sos_token = num_embeddings + 1

        self.precision_dtype = torch.half if precision == "16-mixed" else torch.float32
        self.accelerator = accelerator

        self.cycle_consistency_sigma = cycle_consistency_sigma

        # Loss weights
        self.lcl_weight = latent_consistency_loss_weight
        self.lpips_loss_weight = lpip_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.vq_loss_weight = vq_loss_weight
        self.discriminator_weight = discriminator_weight

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
            image_size,
            patch_size,
            embedding_dim,
            decoder_layer,
            decoder_head,
        )

        # Losses
        if discriminator_type == 'patch-vit':
            self.discriminator = PatchVITDiscriminator(
                image_size=image_size,
                patch_size=patch_size,
                emb_dim=embedding_dim,
                num_layer=disc_num_layers,
                num_head=disc_num_heads,
            )
        else:
            self.discriminator = NLayerDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                use_actnorm=disc_use_actnorm,
                ndf=disc_ndf,
            ).apply(weights_init)

        self.disc_train_steps = disc_train_steps
        self.discriminator_epoch_start = gan_loss_epoch_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")

        self.disc_factor = disc_factor

        self.lpips = lpips.LPIPS(net="vgg")

    def get_cycle_consistency_loss(self, distances, indices):
        q_logits = -1 / 2 * distances / self.cycle_consistency_sigma

        q_logits = q_logits.flatten(0, 1)
        q_indices = indices.flatten()

        # Remove loss for mask token
        mask_token_id = self.feature_quantization.num_embeddings
        q_logits = q_logits[q_indices != mask_token_id]
        q_indices = q_indices[q_indices != mask_token_id]

        return F.cross_entropy(q_logits, q_indices)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight

        return d_weight

    def calculate_discriminator_logits(self, imgs, forward_output, detach=False):
        num_present_images = forward_output.present_forward_indexes.shape[1]

        # Get only present reconstruction images, detach if necessary
        x_recon = imgs[:num_present_images]
        if detach:
            x_recon = x_recon.detach()

        logits = self.discriminator(
            x_recon,
            forward_output.present_forward_indexes,
            forward_output.present_remain_T,
        )
        if forward_output.past_forward_indexes:
            # Get only past reconstructed images, detach if necessary
            x_recon_past = imgs[num_present_images:]
            if detach:
                x_recon_past = x_recon_past.detach()

            past_logits = self.discriminator(
                x_recon_past,
                forward_output.past_forward_indexes,
                forward_output.past_remain_T,
            )
            logits = torch.cat([logits, past_logits])

        return logits

    def discriminator_criterion(
        self,
        forward_output: ForwardOutput,
    ) -> torch.Tensor:
        logits_real = self.calculate_discriminator_logits(
            forward_output.x_target, forward_output, detach=True
        )
        logits_fake = self.calculate_discriminator_logits(
            forward_output.x_recon, forward_output, detach=True
        )

        disc_factor = adopt_weight(
            self.disc_factor,
            self.current_epoch,
            threshold=self.discriminator_epoch_start,
            value=0.0,
        )

        if logits_real.shape[0] and logits_fake.shape[0]:
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
        else:
            d_loss = torch.tensor(0.0, device=self.device)

        return d_loss

    def reconstruction_criterion(
        self,
        forward_output: ForwardOutput,
    ) -> ReconstrcutionCriterionOutput:

        # Compute reconstruction loss
        lpips_loss = self.lpips(forward_output.x_target, forward_output.x_recon).mean()
        l1_loss = torch.abs(forward_output.x_target - forward_output.x_recon).mean()

        reconstruction_loss = (
            l1_loss * self.l1_loss_weight + lpips_loss * self.lpips_loss_weight
        )

        # Compute generator loss
        logits_fake = self.calculate_discriminator_logits(
            forward_output.x_recon, forward_output, detach=False
        )
        generator_loss = -torch.mean(logits_fake)

        try:
            d_weight = self.calculate_adaptive_weight(
                reconstruction_loss, generator_loss, last_layer=self.decoder.head.weight
            )
        except RuntimeError:
            assert not self.training
            d_weight = torch.tensor(0.0)

        disc_factor = adopt_weight(
            self.disc_factor, self.global_step, threshold=self.discriminator_epoch_start
        )

        # Compute consistency loss
        latent_consistency_loss = torch.tensor(0.0, device=self.device)
        if forward_output.past_z_indices is not None:
            latent_consistency_loss += self.get_cycle_consistency_loss(
                forward_output.past_z_distances, forward_output.past_z_indices
            )

        # rescale losses
        reconstruction_loss = reconstruction_loss
        vq_loss = forward_output.vq_loss * self.vq_loss_weight
        generator_loss = generator_loss * d_weight * disc_factor
        latent_consistency_loss = latent_consistency_loss * self.lcl_weight

        # compute final loss
        loss = vq_loss + reconstruction_loss + generator_loss + latent_consistency_loss

        return ReconstrcutionCriterionOutput(
            loss=loss,
            vq_loss=forward_output.vq_loss,
            reconstruction_loss=reconstruction_loss,
            generator_loss=generator_loss,
            latent_consistency_loss=latent_consistency_loss,
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
        (
            fo_masked_features,
            fo_forward_indexes,
            fo_backward_indexes,
            fo_remain_T,
        ) = self.encoder(x)

        # Quantize features
        with torch.autocast(self.accelerator, dtype=torch.float32):
            (
                fo_vq_loss,
                fo_quantized_masked_features,
                fo_perplexity,
                fo_z_indices,
                fo_avg_probs,
                fo_z_distances,
            ) = self.feature_quantization(fo_masked_features, return_distances=True)

        # Reconstruct image from masked input
        fo_z_indices = rearrange(fo_z_indices, "(b t) 1 -> b t", b=x.shape[0])
        x_recon, mask = self.decoder(fo_quantized_masked_features, fo_backward_indexes)

        # Encode reconstructed image again to calculate lcl
        so_masked_features, *_ = self.encoder(x_recon)

        # Quantize features
        with torch.autocast(self.accelerator, dtype=torch.float32):
            (
                so_vq_loss,
                so_quantized_masked_features,
                so_perplexity,
                so_z_indices,
                so_avg_probs,
                so_z_distances,
            ) = self.feature_quantization(so_masked_features, return_distances=True)

        so_z_indices = rearrange(
            so_z_indices, "(b t) 1 -> b t", b=fo_z_indices.shape[0]
        )

        return dict(
            mask=mask,
            fo_masked_features=fo_masked_features,
            fo_quantized_masked_features=fo_quantized_masked_features,
            fo_z_indices=fo_z_indices,
            fo_z_distances=fo_z_distances,
            x_recon=x_recon,
            fo_vq_loss=fo_vq_loss,
            fo_perplexity=fo_perplexity,
            fo_avg_probs=fo_avg_probs,
            # fo masking arguments
            fo_forward_indexes=fo_forward_indexes,
            fo_backward_indexes=fo_backward_indexes,
            fo_remain_T=fo_remain_T,
            # so
            so_masked_features=so_masked_features,
            so_quantized_masked_features=so_quantized_masked_features,
            so_z_indices=so_z_indices,
            so_z_distances=so_z_distances,
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

        (
            so_masked_features,
            so_forward_indexes,
            so_backward_indexes,
            so_remain_T,
        ) = self.encoder(x_recon, ratio=0)

        # Quantize features
        with torch.autocast(self.accelerator, dtype=torch.float32):
            (
                so_vq_loss,
                so_quantized_masked_features,
                so_perplexity,
                so_z_indices,
                so_avg_probs,
                so_z_distances,
            ) = self.feature_quantization(so_masked_features, return_distances=True)

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
            so_vq_loss=so_vq_loss,
            so_perplexity=so_perplexity,
            so_avg_probs=so_avg_probs,
            so_x_recon=so_x_recon,
            # masking arguments
            so_forward_indexes=so_forward_indexes,
            so_backward_indexes=so_backward_indexes,
            so_remain_T=so_remain_T,
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

        present_forward_indexes = present_forward_output["fo_forward_indexes"]
        present_remain_T = present_forward_output["fo_remain_T"]

        past_forward_indexes = None
        past_remain_T = None

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

            # extend vq-vae objectives
            vq_loss = vq_loss + past_forward_output["so_vq_loss"]
            perplexity = perplexity + past_forward_output["so_perplexity"]
            avg_probs = (avg_probs * 2 + past_forward_output["so_avg_probs"]) / 3

            past_forward_indexes = past_forward_output["so_forward_indexes"]
            past_remain_T = past_forward_output["so_remain_T"]

        return ForwardOutput(
            x_recon=x_recon,
            x_target=x_target,
            z_indices=z_indices,
            z_distances=z_distances,
            z_indices_target=z_indices_target,
            past_z_indices=past_z_indices,
            past_z_distances=past_z_distances,
            past_z_indices_target=past_z_indices_target,
            vq_loss=vq_loss,
            perplexity=perplexity,
            avg_probs=avg_probs,
            present_forward_indexes=present_forward_indexes,
            present_remain_T=present_remain_T,
            past_forward_indexes=past_forward_indexes,
            past_remain_T=past_remain_T,
        )

    def training_step(self, batch, batch_idx):
        qmae_opt, d_opt = self.optimizers()

        data, y, *_ = batch
        forward_output = self.forward(data, y)

        # qmae + generator opt step
        criterion_output = self.reconstruction_criterion(forward_output)
        qmae_loss = criterion_output.loss / self.accumulate_batch_every

        self.manual_backward(qmae_loss)

        if (batch_idx + 1) % self.accumulate_batch_every == 0:
            self.clip_gradients(
                d_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )  # better be safe than sorry
            qmae_opt.step()
            qmae_opt.zero_grad()

        # generator opt step
        for _ in range(self.disc_train_steps):
            discriminator_loss = self.discriminator_criterion(forward_output)
            discriminator_loss = discriminator_loss / self.accumulate_batch_every

            self.manual_backward(discriminator_loss)

            self.clip_gradients(
                d_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )  # better be safe than sorry
            d_opt.step()
            d_opt.zero_grad()

        # LOGGING
        self.log_with_postfix(
            "train/loss",
            criterion_output.loss.cpu().item(),
        )
        self.log_with_postfix(
            "train/latent_consistency_loss",
            criterion_output.latent_consistency_loss.cpu().item(),
        )
        self.log_with_postfix(
            "train/generator_loss",
            criterion_output.generator_loss.cpu().item(),
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
            "train/discriminator_loss",
            discriminator_loss.cpu().item(),
        )
        self.log_with_postfix(
            "train/perplexity",
            forward_output.perplexity.cpu().item(),
        )

        return {
            "loss": criterion_output.loss + discriminator_loss,
            "forward_output": forward_output,
        }

    def validation_step(self, batch, batch_idx):
        data, y, *_ = batch

        forward_output = self.forward(data, y)

        criterion_output = self.reconstruction_criterion(forward_output)
        discriminator_loss = self.discriminator_criterion(forward_output)

        # LOGGING
        self.log_with_postfix(
            "val/loss",
            criterion_output.loss.cpu().item(),
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
            "val/latent_consistency_loss",
            criterion_output.latent_consistency_loss.cpu().item(),
        )
        self.log_with_postfix(
            "val/discriminator_loss",
            discriminator_loss.cpu().item(),
        )
        self.log_with_postfix(
            "val/generator_loss",
            criterion_output.generator_loss.cpu().item(),
        )
        self.log_with_postfix(
            "val/perplexity",
            forward_output.perplexity.cpu().item(),
        )

        return {
            "loss": criterion_output.loss + discriminator_loss,
            "forward_output": forward_output,
        }

    def configure_optimizers(self):
        optimizer_qmae = torch.optim.AdamW(
            chain(
                self.encoder.parameters(),
                self.decoder.parameters(),
            ),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )

        optimizer_loss = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.9),
        )

        return [optimizer_qmae, optimizer_loss]

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
