import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

import wandb
from avalanche.core import SupervisedPlugin
from src.avalanche.strategies import NaivePytorchLightning
from src.qmae_latent_extension.model.vit_vq_vae import VitVQVae


class ReconstructionVisualizationPlugin(SupervisedPlugin):
    def __init__(self, num_tasks_in_batch: int):
        super().__init__()

        self.num_tasks_in_batch = num_tasks_in_batch

    @torch.no_grad()
    def before_eval(self, strategy: "NaivePytorchLightning", **kwargs):
        """Update the buffer."""

        model: VitVQVae = strategy.model
        eval_stream = strategy.current_eval_stream

        reconstruction_images = []
        target_images = []

        reconstruction_losses = []
        experience_id = []

        predicted_classes = []
        target_classes = []

        for exp in eval_stream:
            dataloader = DataLoader(
                Subset(exp.dataset, list(range(100))),
                num_workers=0,
                batch_size=32,
                shuffle=False,
            )

            for x, y, _ in dataloader:
                x = x.to(model.device)

                # We shift class w.r.t to experience step due to avalanche return format
                y = y.to(model.device)

                forward_output = model.forward(x)

                # Add images and predicted classes
                reconstruction_images.extend(
                    [
                        wandb.Image(
                            self._rescale_image(forward_output.x_recon[i])
                            .cpu()
                            .numpy(),
                            caption=f"reconstruction_image_{i}",
                        )
                        for i in range(forward_output.x_recon.shape[0])
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
                        F.mse_loss(forward_output.x_recon[i], x[i]).cpu().item()
                        for i in range(x.shape[0])
                    ]
                )

                predicted_classes.extend(
                    forward_output.clf_logits.argmax(dim=-1).cpu().tolist()
                )
                target_classes.extend(y.cpu().tolist())

                experience_id.extend([exp.current_experience] * x.shape[0])

        # Log table with images
        columns = [
            "target",
            "reconstruction",
            "loss",
            "target_class",
            "predicted_class",
            "experience_step",
        ]
        image_data_table = wandb.Table(columns=columns)

        for row in zip(
            target_images,
            reconstruction_images,
            reconstruction_losses,
            target_classes,
            predicted_classes,
            experience_id,
        ):
            image_data_table.add_data(*row)

        wandb.log(
            {
                f"val/reconstructions/experience_step_{strategy.experience_step}": image_data_table
            }
        )

    @staticmethod
    def _rescale_image(image):
        image = torch.clone(image) + 0.5
        image = torch.clamp(image, 0) * 255

        return image.permute(1, 2, 0).int()
