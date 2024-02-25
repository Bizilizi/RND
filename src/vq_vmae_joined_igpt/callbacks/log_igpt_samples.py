import pathlib
import typing as t

import pytorch_lightning as pl
import torch
from avalanche.benchmarks import SplitCIFAR10
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

import wandb
from pytorch_lightning.utilities.types import DistributedDataParallel
from torchvision.utils import make_grid

from src.vq_vmae_joined_igpt.train_image_gpt import sample_images, get_image_embedding


class LogIGPTSamples(Callback):
    def __init__(
        self,
        local_rank: int,
        temperature: float,
        quantize_top_k: int,
        benchmark: SplitCIFAR10,
        log_every=5,
    ):
        super().__init__()
        self.log_every = log_every
        self.local_rank = local_rank
        self.temperature = temperature
        self.quantize_top_k = quantize_top_k
        self.benchmark = benchmark

    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        model: torch.nn.Module,
        unused=None,
    ):
        if trainer.current_epoch % self.log_every != 0:
            return

        model = trainer.model
        if isinstance(model, DistributedDataParallel):
            model = model.module

        experience_step = model.experience_step

        image_embeddings = get_image_embedding(
            vq_vae_model=model,
            num_embeddings=model.num_all_embeddings + 1,
            embedding_dim=model.embedding_dim,
            mask_token=model.mask_token,
        )

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                classes_seen_so_far = self.benchmark.train_stream[
                    experience_step
                ].classes_seen_so_far

                images, *_ = sample_images(
                    image_gpt=model.image_gpt,
                    vq_vae_model=model,
                    embedding=image_embeddings,
                    sos_token=model.sos_token,
                    mask_token=model.mask_token,
                    temperature=self.temperature,
                    max_length=(16 * 16 + 1) * self.quantize_top_k + 1,
                    num_neighbours=self.quantize_top_k,
                    classes_seen_so_far=classes_seen_so_far,
                    num_tokens_without_sos=model.num_all_embeddings + 1,
                )

                sample = make_grid(images.cpu().data)
                sample = (sample + 0.5) * 255
                sample = sample.clip(0, 255)

                # Log sampled images
                logger.log_metrics(
                    {
                        f"train/dataset/experience_step_{experience_step}/igpt_samples": wandb.Image(
                            sample.permute(1, 2, 0).numpy()
                        ),
                        "epoch": trainer.current_epoch,
                    }
                )
