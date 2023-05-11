import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import wandb
from torchvision.utils import make_grid

from src.vq_vae.model.image_gpt_casual import ImageGPTCausal
from src.vq_vae.model.vq_vae import VQVae


class LogIgptSamples(Callback):
    def __init__(
        self,
        vq_vae_model: VQVae,
        experience_step: int,
        num_images: int = 300,
        log_every: int = 10,
    ):
        self.vq_vae_model = vq_vae_model
        self.experience_step = experience_step
        self.num_images = num_images
        self.log_every = log_every

    @torch.no_grad()
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch % self.log_every != 0:
            return

        model: ImageGPTCausal = trainer.model
        self.vq_vae_model.to(model.device)

        for logger in trainer.loggers:
            image = self.get_grid_image(model.image_gpt)
            if isinstance(logger, WandbLogger):
                self.log_wandb(image)
            if isinstance(logger, TensorBoardLogger):
                writer = logger.experiment
                self.log_tb(image, writer, trainer.global_step)

    def get_grid_image(self, igpt_model):
        context = torch.full((self.num_images, 1), 1)  # initialize with SOS token
        context = torch.tensor(context).to(igpt_model.device)
        output = igpt_model.generate(
            input_ids=context,
            max_length=8 * 8 + 1,
            temperature=1.0,
            do_sample=True,
            top_k=40,
        )

        output = output[:, 1:]
        output[output == 512] = 0
        output = output.to(self.vq_vae_model.device)

        quantized = self.vq_vae_model.vq_vae._embedding(output).permute(0, 2, 1)
        quantized = quantized.reshape(-1, quantized.shape[1], 8, 8)

        x_recon = self.vq_vae_model.decoder(quantized)

        grid_image = make_grid(
            x_recon.cpu().data,
        )
        grid_image = self._rescale_image(grid_image)

        return grid_image

    def log_wandb(self, grid_image):
        wandb.log(
            {
                f"train/dataset/experience_step_{self.experience_step}/igpt_samples": wandb.Image(
                    grid_image.cpu().numpy(),
                    caption=f"igpt_samples",
                )
            }
        )

    def log_tb(self, grid_image, writer, step):
        writer.add_image(
            f"train/dataset/experience_step_{self.experience_step}/igpt_samples",
            grid_image.permute(2, 0, 1).cpu().numpy(),
            step,
        )

    @staticmethod
    def _rescale_image(image):
        image = (image + 0.5) * 255
        image = torch.clamp(image, 0, 255)

        return image.permute(1, 2, 0).int()
