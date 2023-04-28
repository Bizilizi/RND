import typing as t
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader

import wandb
from src.vae_ft.model.vae import MLPVae


class LogLatentSpace(Callback):
    def __init__(
        self,
        num_images: int = 200,
    ):
        self.num_images = num_images

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        model: t.Union[MLPVae] = trainer.model
        model.eval()

        z_dim = model.z_dim

        if z_dim != 2:
            return

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                experience_step = model.experience_step
                experience = model.experience

                with torch.no_grad():
                    datasets = [
                        previous_exp.dataset
                        for previous_exp in experience.origin_stream[
                            : experience_step + 1
                        ]
                    ]
                    data_loader = DataLoader(
                        dataset=ConcatDataset(datasets),
                        batch_size=256,
                        shuffle=True,
                        num_workers=2,
                    )

                    x_samples, classes, rec_losses, kl_losses = [], [], [], []
                    x_len = 0

                    for x, y, *_ in data_loader:
                        if x_len >= self.num_images * (experience_step + 1):
                            break

                        x_pred, x, log_sigma, mu = model.forward(x.to(model.device))
                        if not model.decoder.apply_sigmoid:
                            x_pred = torch.sigmoid(x_pred)

                        kl_div = -0.5 * torch.sum(
                            1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=1
                        )
                        reconstruction_loss = F.binary_cross_entropy(
                            x_pred.flatten(1), x.flatten(1), reduction="none"
                        ).sum(dim=1)

                        x_samples.append(x)
                        classes.append(y)
                        kl_losses.append(kl_div)
                        rec_losses.append(reconstruction_loss)

                        x_len += x.shape[0]

                    x_samples = torch.cat(x_samples).to(model.device)

                    projections = model.encoder(x_samples)[0].cpu()
                    classes = torch.cat(classes)
                    kl_losses = torch.cat(kl_losses).cpu()
                    rec_losses = torch.cat(rec_losses).cpu()

                data = [
                    coo + [class_name, rec_l, kl_l]
                    for coo, class_name, rec_l, kl_l in zip(
                        projections.tolist(),
                        classes,
                        rec_losses.tolist(),
                        kl_losses.tolist(),
                    )
                ]

                table = wandb.Table(
                    data=data,
                    columns=["x", "y", "class", "reconstruction loss", "KL loss"],
                )
                scatter_plot = wandb.plot_table(
                    "vgg-continual-learning/scatter/heatmap",
                    table,
                    fields={
                        "x": "x",
                        "y": "y",
                    },
                    string_fields={
                        "title": f"Latent space projections",
                        "groupKeys": "class",
                    },
                )

                wandb.log(
                    {
                        f"train/latent_space/experience_step_{experience_step}/projections": scatter_plot
                    }
                )
