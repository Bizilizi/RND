import umap
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, Subset

import wandb


class VisualizeProjections(Callback):
    """
    This callback visualizes image embeddings in case they have dim = 2
    """

    def __init__(
        self,
        benchmark,
        batch_size=128,
        num_images: int = 100,
        log_every: int = 200,
    ):
        self.benchmark = benchmark
        self.num_images = num_images
        self.batch_size = batch_size
        self.log_every = log_every

    @torch.no_grad()
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch % self.log_every != 0:
            return

        model = trainer.model
        if isinstance(model, DistributedDataParallel):
            model = model.module

        experience_step = model.experience_step

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                # project real dataset
                dataset_to_project = ConcatDataset(
                    [
                        experience.dataset
                        for experience in self.benchmark.train_stream[
                            : experience_step + 1
                        ]
                    ]
                )

                real_image_embs, real_classes = self.project_dataset(
                    model, dataset_to_project
                )

                # Log real dataset
                real_data = torch.cat(
                    [
                        real_image_embs,
                        real_classes[..., None],
                        torch.ones(real_image_embs.shape[0], 1),  # is real flag
                    ],
                    dim=1,
                )
                wandb.log(
                    {
                        f"train/projections/real_data_experience_step_{experience_step}": wandb.Table(
                            columns=["x", "y", "class", "is_real"],
                            data=real_data.tolist(),
                        )
                    }
                )

                # project bootstrapped dataset
                if experience_step > 0:
                    dataset_to_project = trainer.datamodule.train_dataset
                    (
                        bootstrapped_image_embs,
                        bootstrapped_classes,
                    ) = self.project_dataset(model, dataset_to_project)

                    # Log bootstrapped dataset
                    bootstrapped_data = torch.cat(
                        [
                            bootstrapped_image_embs,
                            bootstrapped_classes[..., None],
                            torch.zeros(
                                bootstrapped_image_embs.shape[0], 1
                            ),  # is real flag
                        ],
                        dim=1,
                    )
                    wandb.log(
                        {
                            f"train/projections/bootstrapped_data_experience_step_{experience_step}": wandb.Table(
                                columns=["x", "y", "class", "is_real"],
                                data=bootstrapped_data.tolist(),
                            )
                        }
                    )
                else:
                    bootstrapped_data = torch.zeros((0, 4))

                # Log joined real and bootstrapped datasets
                data = torch.cat([real_data, bootstrapped_data], dim=0).tolist()
                data = [
                    [
                        x,
                        y,
                        int(class_id),
                        int(is_real),
                        f"{int(class_id)}_{'r' if is_real else 'b'}",
                    ]
                    for x, y, class_id, is_real in data
                ]

                wandb.log(
                    {
                        f"train/projections/all_data_experience_step_{experience_step}": wandb.Table(
                            columns=["x", "y", "class", "is_real", "full_class"],
                            data=data,
                        )
                    }
                )

    def project_dataset(self, model, dataset_to_project):
        experience_step = model.experience_step
        dataloader = DataLoader(
            dataset_to_project,
            num_workers=8,
            batch_size=self.batch_size,
            shuffle=True,
        )

        image_embs = []
        classes = []

        count = 0
        for x, y, _ in dataloader:
            if count >= self.num_images * (experience_step + 1):
                break

            if isinstance(x, dict):
                # Project only past bootstrapped data
                past_data_mask = x["is_past_domain"] == 1

                if past_data_mask.any():
                    x = x["images"][past_data_mask]
                    y = y[past_data_mask]
                else:
                    continue

            x = x.to(model.device)
            _, full_features, _ = model.encoder(x, return_full_features=True)
            image_emb = model.get_image_embedding(full_features)

            image_embs.append(image_emb)
            classes.append(y)

            count += x.shape[0]

        image_embs = torch.cat(image_embs).cpu()
        classes = torch.cat(classes).cpu()

        if image_embs.shape[-1] != 2:
            n_neighbors = min(15, image_embs.shape[0] // 2)
            image_embs = umap.UMAP(n_neighbors=n_neighbors).fit_transform(image_embs)
            image_embs = torch.tensor(image_embs)

        return image_embs, classes
