import datetime
import os
import pathlib
import shutil
from configparser import ConfigParser
import typing as t
import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import wandb
from avalanche.benchmarks import SplitCIFAR10
from torchvision.utils import make_grid
from tqdm import trange, tqdm
from transformers import ImageGPTConfig

from src.avalanche.strategies import NaivePytorchLightning
from src.transformer_vq_vae.callbacks.reconstruction_visualization_plugin import (
    ReconstructionVisualizationPlugin,
)
from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.data.image_gpt_dataset import ImageGPTDataset
from src.transformer_vq_vae.init_scrips import (
    get_callbacks,
    get_evaluation_plugin,
    get_model,
    get_train_plugins,
)
from src.transformer_vq_vae.model.image_gpt import ImageGPTForCausalImageModeling
from src.transformer_vq_vae.model_future import model_future_samples
from src.transformer_vq_vae.train_classifier import (
    train_classifier_on_all_classes,
    train_classifier_on_observed_only_classes,
)
from src.transformer_vq_vae.train_image_gpt import (
    bootstrap_past_samples,
    train_igpt,
    init_token_embeddings,
    get_image_embedding,
    learning_rate_schedule,
    sample_images,
)
from src.transformer_vq_vae.utils.wrap_empty_indices import (
    convert_avalanche_dataset_to_vq_mae_dataset,
)
from src.utils.summary_table import log_summary_table_to_wandb
from src.utils.train_script import overwrite_config_with_args
from train_utils import get_device, get_loggers, get_wandb_params

from pathlib import Path


def get_num_random_past_samples(
    config: TrainConfig, cl_strategy: NaivePytorchLightning
):
    if config.num_random_past_samples_schedule == "fixed":
        return config.num_random_past_samples

    if config.num_random_past_samples_schedule == "linear":
        return config.num_random_past_samples * cl_strategy.experience_step

    if config.num_random_past_samples_schedule == "schedule":
        schedule = torch.linspace(0, config.num_random_past_samples, config.num_tasks)
        return int(schedule[int(cl_strategy.experience_step)])


def mock_train_igpt(
    strategy: NaivePytorchLightning,
    config: TrainConfig,
    train_dataset: Dataset,
    device: torch.device,
    classes_seen_so_far: t.List[int],
    num_all_classes: int,
    n_layer: int = 12,
    image_gpt: ImageGPTForCausalImageModeling = None,
):
    vq_vae_model = strategy.model
    logger = strategy.train_logger

    vocab_size = (
        vq_vae_model.feature_quantization.embedding.num_embeddings + 2 + num_all_classes
    )
    """
    number of embeddings in codebook 
    +
    mask_token 
    + 
    sos_token
    +
    num of all classes 
    """
    mask_token = vq_vae_model.feature_quantization.embedding.num_embeddings
    sos_token = vq_vae_model.feature_quantization.embedding.num_embeddings + 1

    # Derive num patches based on path algorithm from VIT
    num_patches = (config.image_size // config.patch_size) ** 2

    configuration = ImageGPTConfig(
        **{
            "activation_function": "quick_gelu",
            "attn_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-05,
            "model_type": "imagegpt",
            "n_embd": config.embedding_dim,
            "n_head": 8,
            "n_layer": n_layer,
            "n_positions": (num_patches + 1) * config.quantize_top_k + 2,
            "reorder_and_upcast_attn": False,
            "resid_pdrop": 0.1,
            "scale_attn_by_inverse_layer_idx": False,
            "scale_attn_weights": True,
            "tie_word_embeddings": False,
            "use_cache": False,
            "vocab_size": vocab_size,
        }
    )
    image_gpt = image_gpt or ImageGPTForCausalImageModeling(configuration)

    init_token_embeddings(vq_vae_model, image_gpt, config, mask_token)
    image_embeddings = get_image_embedding(vq_vae_model, config, mask_token).to(
        vq_vae_model.device
    )

    vq_vae_model.to(device)
    image_gpt.to(device)
    image_embeddings.to(device)

    train_dataset = ImageGPTDataset(
        vq_vae_model=vq_vae_model,
        dataset=train_dataset,
        sos_token=sos_token,
        mask_token=mask_token,
        ratio=config.igpt_mask_ratio,
        num_workers=config.num_workers,
        top_k=config.quantize_top_k,
        supervised=config.supervised,
    )
    data_loader = DataLoader(
        train_dataset,
        batch_size=config.igpt_batch_size,
        shuffle=True,
    )

    if strategy.experience_step < 2:
        epoch_num = 1
    elif strategy.experience_step < 3:
        epoch_num = 1
    else:
        epoch_num = 1

    grad_scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(image_gpt.parameters(), lr=3e-3)
    exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_schedule(
            500, epoch_num * len(data_loader) // config.igpt_accumulate_grad_batches
        ),
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    step = 0
    for i in trange(0, epoch_num):
        counter = i
        logger.log_metrics({"igpt_epoch": counter}, step=step)

        for batch in tqdm(data_loader):
            step += 1

            input_ids = batch["masked_input_ids"].to(device)
            # with torch.autocast(device_type=config.accelerator):
            output = image_gpt(input_ids=input_ids)
            loss = loss_fn(
                output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
                input_ids[..., 1:].reshape(-1),
            )
            # grad_scaler.scale(loss).backward()

            if step % config.igpt_accumulate_grad_batches == 0:
                # grad_scaler.step(optimizer)
                # grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                exp_lr_scheduler.step()

            logger.log_metrics(
                {
                    f"train/image_gpt_loss/experience_step_{strategy.experience_step}": loss,
                    "epoch": i,
                },
                step=i,
            )

            if step % 500 == 0:

                images, *_ = sample_images(
                    image_gpt=image_gpt,
                    vq_vae_model=vq_vae_model,
                    embedding=image_embeddings,
                    sos_token=sos_token,
                    temperature=config.temperature,
                    max_length=(num_patches + 1) * config.quantize_top_k + 1,
                    num_neighbours=config.quantize_top_k,
                    supervised=config.supervised,
                    classes_seen_so_far=classes_seen_so_far,
                )

                sample = make_grid(images.cpu().data)
                sample = (sample + 0.5) * 255
                sample = sample.clip(0, 255)

                if isinstance(logger, WandbLogger):
                    logger.log_metrics(
                        {
                            f"train/dataset/experience_step_{strategy.experience_step}/igpt_samples": wandb.Image(
                                sample.permute(1, 2, 0).numpy()
                            ),
                            "step": step,
                        }
                    )
                if isinstance(logger, TensorBoardLogger):
                    logger.experiment.add_image(
                        f"train/dataset/experience_step_{strategy.experience_step}/igpt_samples",
                        sample / 255,
                        step,
                    )

            if step % 100 == 0:
                model_ckpt_path = f"{config.checkpoint_path}/igpt-exp{strategy.experience_step}-{i}.ckpt"
                state_dict = image_gpt.state_dict()
                for k, v in state_dict.items():
                    state_dict[k] = v.cpu()

                torch.save(state_dict, model_ckpt_path)

    return image_gpt


def mock_train_loop(
    benchmark: SplitCIFAR10,
    cl_strategy: NaivePytorchLightning,
    is_using_wandb: bool,
    config: TrainConfig,
    device: torch.device,
) -> None:
    """
    :return:
    """

    image_gpt = None

    for train_experience, test_experience in zip(
        benchmark.train_stream, benchmark.test_stream
    ):

        train_experience.dataset = convert_avalanche_dataset_to_vq_mae_dataset(
            train_experience.dataset,
            num_neighbours=config.quantize_top_k,
            config=config,
            time_tag=0,
        )
        test_experience.dataset = convert_avalanche_dataset_to_vq_mae_dataset(
            test_experience.dataset,
            num_neighbours=config.quantize_top_k,
            config=config,
            time_tag=0,
        )
        igpt_train_dataset = train_experience.dataset + test_experience.dataset

        # Bootstrap old data and modeled future samples
        if cl_strategy.experience_step != 0 and image_gpt is not None:
            image_gpt.to(device)
            cl_strategy.model.to(device)

            if config.num_random_past_samples != 0:
                print(f"Bootstrap vae model..")
                previous_classes = list(
                    set(train_experience.classes_seen_so_far).difference(
                        train_experience.classes_in_this_experience
                    )
                )
                bootstrapped_dataset = bootstrap_past_samples(
                    image_gpt=image_gpt,
                    vq_vae_model=cl_strategy.model,
                    num_images=get_num_random_past_samples(config, cl_strategy),
                    dataset_path=config.bootstrapped_dataset_path,
                    config=config,
                    experience_step=cl_strategy.experience_step,
                    classes_seen_so_far=previous_classes,
                )

                train_experience.dataset = (
                    train_experience.dataset + bootstrapped_dataset
                )

                igpt_train_dataset = igpt_train_dataset + bootstrapped_dataset

            if config.num_random_future_samples != 0:
                print(f"Model future samples..")
                future_dataset = model_future_samples(
                    vq_vae_model=cl_strategy.model,
                    num_images=(
                        config.num_random_future_samples
                        * (4 - cl_strategy.experience_step)
                    ),
                    mode=config.future_samples_mode,
                    config=config,
                )

                train_experience.dataset = train_experience.dataset + future_dataset

        # Train VQ-VAE
        # cl_strategy.train(train_experience, [test_experience])

        # Train linear classifier, but before we freeze model params
        # We train two classifiers. One to predict all classes,
        # another to predict only observed so far classes.
        # cl_strategy.model.freeze()

        # Train classifier
        # print(f"Train classifier..")
        # all_clf_head = train_classifier_on_all_classes(
        #     strategy=cl_strategy, config=config, benchmark=benchmark, device=device
        # ).to(device)
        # train_classifier_on_observed_only_classes(
        #     strategy=cl_strategy, config=config, benchmark=benchmark, device=device
        # ).to(device)

        # cl_strategy.model.set_clf_head(all_clf_head)

        # Train new image gpt model
        print(f"Train igpt..")
        image_gpt = mock_train_igpt(
            strategy=cl_strategy,
            config=config,
            train_dataset=igpt_train_dataset,
            device=device,
            n_layer=config.num_gpt_layers,
            classes_seen_so_far=train_experience.classes_seen_so_far,
            num_all_classes=benchmark.n_classes,
        )

        # Evaluate VQ-VAE and linear classifier
        # cl_strategy.eval(benchmark.test_stream)

        # Reset linear classifier and unfreeze params
        # cl_strategy.model.reset_clf_head()

        cl_strategy.model.unfreeze()
        cl_strategy.experience_step += 1

    if is_using_wandb:
        log_summary_table_to_wandb(benchmark.train_stream, benchmark.test_stream)
