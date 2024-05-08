import argparse
import pathlib
from configparser import ConfigParser

import wandb
import typing as t

from avalanche.benchmarks.utils.classification_dataset import (
    ClassificationDataset,
    make_classification_dataset,
)
from torch.utils.data import Dataset, ConcatDataset
import torch
from tqdm.auto import trange
from transformers import ImageGPTConfig

from src.image_gpt.configuration.config import TrainConfig
from src.image_gpt.init_scrips import (
    get_benchmark,
    get_model,
    get_cl_strategy,
)
from src.image_gpt.model.image_gpt import ImageGPTForCausalImageModeling
from src.image_gpt.train_image_gpt import (
    BootstrappedDataset,
    get_image_embedding,
    sample_images,
    bootstrap_past_samples,
)
from src.image_gpt.utils.fid_score import calculate_fid_given_datasets
from src.image_gpt.utils.wrap_empty_indices import (
    wrap_dataset,
)
from train_utils import get_device


class ImgDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, *_ = self.dataset[item]
        return x["images"]


def calculate_fid_score(
    *,
    config,
    benchmark,
    model,
    image_gpt,
    device,
    run_id,
    exp_step,
    task_id=None,
    m_ep,
    i_ep,
    num_images=1000,
):
    ## WARNING: BER CAREFUL WITH CALCULATION OF THESE
    ## HAVE TO BE CONSISTENT WITH TRAINING SCRIPT
    sos_token = config.num_embeddings + 1
    mask_token = config.num_embeddings

    model.load_state_dict(
        torch.load(
            f"/scratch/shared/beegfs/dzverev/artifacts/{run_id}/model/model-exp-{exp_step}-ep-{m_ep}.ckpt",
            map_location=torch.device("cpu"),
        )["state_dict"]
    )
    image_gpt.load_state_dict(
        torch.load(
            f"/scratch/shared/beegfs/dzverev/artifacts/{run_id}/model/igpt-exp{exp_step}-{i_ep}.ckpt",
            map_location=torch.device("cpu"),
        )
    )

    model.to(device)
    image_gpt.to(device)

    # create datasets
    """
    If task_id is empty we calculate fid score for all tasks
    observed before given experience step
    """
    bootstrapped_dataset = bootstrap_past_samples(
        image_gpt=image_gpt,
        qmae_model=model,
        num_images=num_images,
        classes_seen_in_past=benchmark.train_stream[exp_step].classes_seen_so_far,
        config=config,
    )
    bootstrapped_dataset = ImgDataset(bootstrapped_dataset)

    real_dataset = ConcatDataset(
        [
            wrap_dataset(
                experience.dataset,
                img_embedding_dim=config.img_embedding_dim,
                is_past_domain=True,
            )
            for experience_step, experience in enumerate(benchmark.train_stream)
        ]
    )
    real_dataset = ImgDataset(real_dataset)

    return calculate_fid_given_datasets(
        bootstrapped_dataset, real_dataset, 128, device, 2048
    )


def calculate_fid_score_for_all_cl_steps(
    run_id, num_images, max_epochs, min_epochs, i_ep=9
):
    ini_config = ConfigParser()
    ini_config.read("./src/image_gpt/configuration/train.ini")

    config = TrainConfig.construct_typed_config(ini_config)

    run = wandb.init(
        id=run_id,
        project="qmae-latent-extension",
        entity="vgg-continual-learning",
        resume="must",
    )
    config.__dict__.update(run.config)

    config.train_logger = "tensorboard"
    config.evaluation_logger = "int"

    local_rank = 0

    target_dataset_dir = pathlib.Path(f"/tmp/dzverev_data/{config.dataset}")
    device = get_device(config, local_rank)

    benchmark = get_benchmark(config, target_dataset_dir)
    model = get_model(config, device)

    vocab_size = config.num_embeddings + 2 + benchmark.n_classes
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
            "n_layer": 12,
            "n_positions": 16 * 16 + 3,
            "reorder_and_upcast_attn": False,
            "resid_pdrop": 0.1,
            "scale_attn_by_inverse_layer_idx": False,
            "scale_attn_weights": True,
            "tie_word_embeddings": False,
            "use_cache": False,
            "vocab_size": vocab_size,
        }
    )
    image_gpt = ImageGPTForCausalImageModeling(configuration)

    model.to(device)
    image_gpt.to(device)

    fid_scores = []
    m_eps = [max_epochs, min_epochs, min_epochs, min_epochs, min_epochs]

    print("Compute score for all tasks")
    for experience_step in trange(len(benchmark.train_stream)):
        fid_scores.append(
            calculate_fid_score(
                config=config,
                benchmark=benchmark,
                model=model,
                image_gpt=image_gpt,
                device=device,
                run_id=run_id,
                exp_step=experience_step,
                m_ep=m_eps[experience_step],
                i_ep=i_ep,
                num_images=num_images,
            )
        )

    # log to wandb
    wandb.log(
        {
            "fid_score/all_tasks": wandb.Table(
                columns=["experience_step", "value"], data=list(enumerate(fid_scores))
            )
        }
    )

    wandb.finish()
