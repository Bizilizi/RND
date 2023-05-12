import datetime

import torch
import typing as t

import wandb
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
from transformers import ImageGPTConfig, ImageGPTForCausalImageModeling

from src.avalanche.data import PLDataModule
from src.avalanche.strategies import NaivePytorchLightning
from src.rnd.callbacks.log_model import LogModelWightsCallback
from src.vq_vae.callbacks.igpt_samples import LogIgptSamples
from src.vq_vae.configuration.config import TrainConfig
from src.vq_vae.data.image_gpt_dataset import ImageGPTDataset
from src.vq_vae.model.image_gpt_casual import ImageGPTCausal
from src.vq_vae.model.vq_vae import VQVae
import numpy as np
from torch.nn import functional as F
from torchvision.utils import save_image, make_grid
from torchvision.io import read_image


class BootstrappedDataset(Dataset):
    def __init__(self, dataset_path: str, experience_step: int):
        super().__init__()

        self.dataset_path = dataset_path
        self.experience_step = experience_step
        self.x = []
        self.targets = []

    def add_images(self, images):
        total_img = len(self.x)
        for i, image in enumerate(images):
            image_path = (
                f"{self.dataset_path}/exp_{self.experience_step}/{total_img + i}.png"
            )
            save_image(self._rescale_image(image), image_path)

            self.x.append(image_path)
            self.targets.append(-1)

    @staticmethod
    def _rescale_image(image):
        image = (image + 0.5) * 255
        image = torch.clamp(image, 0, 255)

        return image.int()

    def __getitem__(self, item):
        return read_image(self.x[item]), self.targets[item]

    def __len__(self):
        return len(self.x)


@torch.no_grad()
def bootstrap_dataset(
    image_gpt: ImageGPTCausal,
    vq_vae_model: VQVae,
    num_images: int,
    experience_step: int,
    dataset_path: str,
) -> ClassificationDataset:
    images = []
    num_images_per_batch = min(128, num_images)
    bootstrapped_dataset = BootstrappedDataset(
        dataset_path=dataset_path, experience_step=experience_step
    )

    for _ in range(num_images // num_images_per_batch):
        context = torch.full((num_images_per_batch, 1), 1)  # initialize with SOS token
        context = torch.tensor(context).to(image_gpt.device)
        output = image_gpt.image_gpt.generate(
            input_ids=context,
            max_length=8 * 8 + 1,
            temperature=1.0,
            do_sample=True,
            top_k=40,
        )

        output = output[:, 1:]
        output[output == 512] = 0

        quantized = vq_vae_model.vq_vae._embedding(output).permute(0, 2, 1)
        quantized = quantized.reshape(-1, quantized.shape[1], 8, 8)

        recon = vq_vae_model.decoder(quantized)
        bootstrapped_dataset.add_images(recon.cpu())

    dataset = make_classification_dataset(
        bootstrapped_dataset, targets=bootstrapped_dataset.targets
    )

    return dataset


# def train_igpt(
#     strategy: NaivePytorchLightning,
#     config: TrainConfig,
#     train_dataset: Dataset,
#     test_dataset: Dataset,
#     device: torch.device,
#     wandb_params: t.Dict[str, t.Any],
# ):
#     today = datetime.datetime.now()
#     run_id = wandb_params["id"] if wandb_params else today.strftime("%Y_%m_%d_%H_%M")
#
#     model = strategy.model.to(device)
#
#     configuration = ImageGPTConfig(
#         activation_function="quick_gelu",
#         attn_pdrop=0.1,
#         embd_pdrop=0.1,
#         initializer_range=0.02,
#         layer_norm_epsilon=1e-05,
#         model_type="imagegpt",
#         n_embd=config.embedding_dim,
#         n_head=4,
#         n_layer=12,
#         n_positions=8 * 8 + 1,
#         reorder_and_upcast_attn=False,
#         resid_pdrop=0.1,
#         scale_attn_by_inverse_layer_idx=False,
#         scale_attn_weights=True,
#         tie_word_embeddings=False,
#         use_cache=False,
#         vocab_size=config.num_embeddings + 1,
#     )
#     image_gpt = ImageGPTCausal(
#         configuration, vq_vae=model, experience_step=strategy.experience_step
#     )
#
#     train_dataset = ImageGPTDataset(
#         vq_vae_model=model,
#         dataset=train_dataset,
#         sos_token=config.num_embeddings,
#     )
#     test_dataset = ImageGPTDataset(
#         vq_vae_model=model,
#         dataset=test_dataset,
#         sos_token=config.num_embeddings,
#     )
#
#     datamodule = PLDataModule(
#         batch_size=512,
#         num_workers=config.num_workers,
#         train_dataset=train_dataset,
#         val_dataset=test_dataset,
#     )
#
#     # Training
#     trainer = Trainer(
#         check_val_every_n_epoch=strategy.validate_every_n,
#         accelerator=strategy.accelerator,
#         devices=strategy.devices,
#         logger=strategy.train_logger,
#         callbacks=[
#             LogIgptSamples(
#                 vq_vae_model=model,
#                 experience_step=strategy.experience_step,
#             ),
#             LogModelWightsCallback(
#                 log_every=10,
#                 checkpoint_path=f"{config.checkpoint_path}/{run_id}",
#                 model_prefix="igpt",
#             ),
#         ],
#         max_epochs=config.max_epochs_igpt,
#         min_epochs=config.min_epochs_igpt,
#     )
#
#     trainer.fit(image_gpt, datamodule=datamodule)
#
#     return image_gpt


def train_igpt(
    strategy: NaivePytorchLightning,
    config: TrainConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
):
    configuration = ImageGPTConfig(
        **{
            "activation_function": "quick_gelu",
            "attn_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-05,
            "model_type": "imagegpt",
            "n_embd": config.embedding_dim,
            "n_head": 4,
            "n_layer": 24,
            "n_positions": 8 * 8 + 1,
            "reorder_and_upcast_attn": False,
            "resid_pdrop": 0.1,
            "scale_attn_by_inverse_layer_idx": False,
            "scale_attn_weights": True,
            "tie_word_embeddings": False,
            "use_cache": False,
            "vocab_size": config.num_embeddings + 1,
        }
    )
    image_gpt = ImageGPTForCausalImageModeling(configuration)
    image_gpt.transformer.wte.weight.data[
        :-1
    ] = strategy.model.vq_vae._embedding.weight.data

    vq_vae_model = strategy.model
    logger = strategy.train_logger
    device = strategy.model.device

    train_dataset = ImageGPTDataset(
        vq_vae_model=vq_vae_model,
        dataset=train_dataset,
        sos_token=config.num_embeddings,
    )
    test_dataset = ImageGPTDataset(
        vq_vae_model=vq_vae_model,
        dataset=test_dataset,
        sos_token=config.num_embeddings,
    )
    data_loader = DataLoader(
        train_dataset,
        batch_size=256,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(
        [
            {"params": list(image_gpt.parameters())[1:], "lr": 0.001},
            {"params": image_gpt.transformer.wte.parameters(), "lr": 0.0001},
        ]
    )
    exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1000,
        eta_min=0.001 * 1e-3,
    )

    vq_vae_model.to(device)
    image_gpt.to(device)

    for i in trange(0, config.max_epochs_igpt, leave=False):

        epoch_losses = []
        for batch in tqdm(data_loader, leave=False):
            optimizer.zero_grad()
            batch["input_ids"] = batch["input_ids"].to(device)

            output = image_gpt(input_ids=batch["input_ids"])
            loss = F.cross_entropy(
                output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
                batch["input_ids"][..., 1:].reshape(-1),
            )
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.cpu().item())

        logger.log_metrics(
            {
                f"train/image_gpt_loss/experience_step_{strategy.experience_step}": np.mean(
                    epoch_losses
                ),
                "epoch": i,
            },
            step=i,
        )

        if i % 2 == 0:
            epoch_losses = []
            with torch.no_grad():
                for batch in tqdm(test_data_loader, leave=False):
                    batch["input_ids"] = batch["input_ids"].to(device)

                    output = image_gpt(input_ids=batch["input_ids"])
                    loss = F.cross_entropy(
                        output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
                        batch["input_ids"][..., 1:].reshape(-1),
                    )
                    epoch_losses.append(loss.cpu().item())

            logger.log_metrics(
                {
                    f"val/image_gpt_loss/experience_step_{strategy.experience_step}": np.mean(
                        epoch_losses
                    ),
                    "epoch": i,
                },
                step=i,
            )

            sample = get_sample_image(image_gpt, vq_vae_model).cpu()
            if isinstance(logger, WandbLogger):
                logger.log_metrics(
                    {
                        f"train/dataset/experience_step_{strategy.experience_step}/igpt_samples": wandb.Image(
                            sample.permute(1, 2, 0).numpy()
                        ),
                        "epoch": i,
                    }
                )
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(
                    f"train/dataset/experience_step_{strategy.experience_step}/igpt_samples",
                    sample / 255,
                    i,
                )

            image_gpt.to(device)
            vq_vae_model.to(device)

        if i % 10 == 0:
            model_ckpt_path = f"{config.checkpoint_path}/igpt-{i}.ckpt"
            state_dict = image_gpt.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()

            torch.save(state_dict, model_ckpt_path)

        exp_lr_scheduler.step()

    igpt = ImageGPTCausal(
        configuration=configuration,
        vq_vae=vq_vae_model,
        experience_step=strategy.experience_step,
    )
    igpt.image_gpt = image_gpt

    return igpt


def get_sample_image(image_gpt, vq_vae_model, num_images=8 * 4 * 10):
    with torch.no_grad():
        context = torch.full((num_images, 1), 1)  # initialize with SOS token
        context = torch.tensor(context).to(image_gpt.device)
        output = image_gpt.generate(
            input_ids=context,
            max_length=8 * 8 + 1,
            temperature=1.0,
            do_sample=True,
            top_k=40,
        )

        output = output[:, 1:]
        output[output == 512] = 0

        quantized = vq_vae_model.vq_vae._embedding(output).permute(0, 2, 1)
        quantized = quantized.reshape(-1, quantized.shape[1], 8, 8)

        x_recon = vq_vae_model.decoder(quantized)

    grid_image = make_grid(x_recon.cpu())
    grid_image = (grid_image + 0.5) * 255
    grid_image = grid_image.clip(0, 255)

    return grid_image
