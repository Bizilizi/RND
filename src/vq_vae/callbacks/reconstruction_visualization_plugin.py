import torch
import wandb
from torch.nn import functional as F
from avalanche.core import SupervisedPlugin
from torch.utils.data import DataLoader, Subset

from src.avalanche.strategies import NaivePytorchLightning
from src.vq_vae.model.vq_vae import VQVae


class ReconstructionVisualizationPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def before_eval(self, strategy: "NaivePytorchLightning", **kwargs):
        """Update the buffer."""

        model: VQVae = strategy.model
        eval_stream = strategy.current_eval_stream

        reconstruction_images = []
        target_images = []
        reconstruction_losses = []
        experience_id = []

        for exp in eval_stream:
            dataloader = DataLoader(
                Subset(exp.dataset, list(range(100))),
                num_workers=1,
                batch_size=32,
                shuffle=False,
            )

            for x, y, _ in dataloader:
                x = x.to(model.device)

                vq_loss, x_recon, quantized, _, perplexity, logits = model.forward(x)
                _, reconstruction_loss, _ = model.criterion(
                    (vq_loss, x_recon, quantized, x, perplexity, logits), y
                )

                reconstruction_images.extend(
                    [
                        wandb.Image(
                            self._rescale_image(x_recon[i]).cpu().numpy(),
                            caption=f"reconstruction_image_{i}",
                        )
                        for i in range(x_recon.shape[0])
                    ]
                )

                target_images.extend(
                    [
                        wandb.Image(
                            self._rescale_image(x[i]).cpu().numpy(),
                            caption=f"reconstruction_image_{i}",
                        )
                        for i in range(x.shape[0])
                    ]
                )

                reconstruction_losses.extend(
                    [
                        F.mse_loss(x_recon[i], x[i]).cpu().item()
                        for i in range(x.shape[0])
                    ]
                )

                experience_id.extend([exp.current_experience] * x.shape[0])

        # Log table with images
        columns = ["target", "reconstruction", "loss", "experience_step"]
        image_data_table = wandb.Table(columns=columns)

        for rec_image, target_image, loss, exp_id in zip(
            reconstruction_images, target_images, reconstruction_losses, experience_id
        ):
            image_data_table.add_data(target_image, rec_image, loss, exp_id)

        wandb.log(
            {
                f"val/reconstructions/experience_step_{strategy.experience_step - 1}": image_data_table
            }
        )

    @staticmethod
    def _rescale_image(image):
        image = torch.clone(image) + 0.5
        image = torch.clamp(image, 0) * 255

        return image.permute(1, 2, 0).int()
