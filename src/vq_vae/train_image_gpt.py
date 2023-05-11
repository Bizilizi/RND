import datetime

import torch
import typing as t
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import Dataset
from transformers import ImageGPTConfig

from src.avalanche.data import PLDataModule
from src.avalanche.strategies import NaivePytorchLightning
from src.rnd.callbacks.log_model import LogModelWightsCallback
from src.vq_vae.callbacks.igpt_samples import LogIgptSamples
from src.vq_vae.configuration.config import TrainConfig
from src.vq_vae.data.image_gpt_dataset import ImageGPTDataset
from src.vq_vae.model.image_gpt_casual import ImageGPTCausal
from src.vq_vae.model.vq_vae import VQVae


class TensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: t.List[int]):
        super().__init__()

        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


@torch.no_grad()
def bootstrap_dataset(
    image_gpt: ImageGPTCausal,
    vq_vae_model: VQVae,
    num_images: int,
) -> ClassificationDataset:
    images = []
    num_images_per_batch = min(128, num_images)

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
        images.append(recon.cpu())

    images = torch.cat(images)
    targets = [-1] * images.shape[0]
    dataset = make_classification_dataset(
        TensorDataset(images, targets), targets=targets
    )

    return dataset


def train_igpt(
    strategy: NaivePytorchLightning,
    config: TrainConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
    device: torch.device,
    wandb_params: t.Dict[str, t.Any],
):
    today = datetime.datetime.now()
    run_id = wandb_params["id"] if wandb_params else today.strftime("%Y_%m_%d_%H_%M")

    model = strategy.model.to(device)

    configuration = ImageGPTConfig(
        activation_function="quick_gelu",
        attn_pdrop=0.1,
        embd_pdrop=0.1,
        initializer_range=0.02,
        layer_norm_epsilon=1e-05,
        model_type="imagegpt",
        n_embd=config.embedding_dim,
        n_head=4,
        n_layer=12,
        n_positions=8 * 8 + 1,
        reorder_and_upcast_attn=False,
        resid_pdrop=0.1,
        scale_attn_by_inverse_layer_idx=False,
        scale_attn_weights=True,
        tie_word_embeddings=False,
        use_cache=False,
        vocab_size=config.num_embeddings + 1,
    )
    image_gpt = ImageGPTCausal(
        configuration, vq_vae=model, experience_step=strategy.experience_step
    )

    train_dataset = ImageGPTDataset(
        vq_vae_model=model,
        dataset=train_dataset,
        sos_token=config.num_embeddings,
    )
    test_dataset = ImageGPTDataset(
        vq_vae_model=model,
        dataset=test_dataset,
        sos_token=config.num_embeddings,
    )

    datamodule = PLDataModule(
        batch_size=512,
        num_workers=config.num_workers,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )

    # Training
    trainer = Trainer(
        check_val_every_n_epoch=strategy.validate_every_n,
        accelerator=strategy.accelerator,
        devices=strategy.devices,
        logger=strategy.train_logger,
        callbacks=[
            LogIgptSamples(
                vq_vae_model=model,
                experience_step=strategy.experience_step,
            ),
            LogModelWightsCallback(
                log_every=10,
                checkpoint_path=f"{config.checkpoint_path}/{run_id}",
                model_prefix="igpt",
            ),
        ],
        max_epochs=config.max_epochs_igpt,
        min_epochs=config.min_epochs_igpt,
    )

    trainer.fit(image_gpt, datamodule=datamodule)

    return image_gpt
