import argparse
import pathlib
from configparser import ConfigParser

import wandb
import typing as t

from avalanche.benchmarks.utils.classification_dataset import (
    ClassificationDataset,
    make_classification_dataset,
)
from einops import rearrange
from torch.utils.data import Dataset, ConcatDataset
import torch
from torchvision.utils import make_grid
from tqdm.auto import trange
from transformers import ImageGPTConfig

from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.init_scrips import get_benchmark, get_model, get_cl_strategy
from src.transformer_vq_vae.model.image_gpt import ImageGPTForCausalImageModeling
from src.transformer_vq_vae.train_image_gpt import (
    BootstrappedDataset,
    get_image_embedding,
)
from src.transformer_vq_vae.utils.fid_score import calculate_fid_given_datasets
from src.transformer_vq_vae.utils.wrap_empty_indices import (
    wrap_dataset_with_empty_indices,
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


@torch.no_grad()
def sample_images(
    image_gpt,
    vq_vae_model,
    embedding,
    sos_token,
    temperature=1.23,
    num_images=8 * 4 * 10,
    return_grid_only=False,
):
    image_gpt.eval()
    vq_vae_model.eval()

    device = vq_vae_model.device
    decoder = vq_vae_model.decoder

    context = torch.full(
        (num_images, 1), sos_token, device=device
    )  # initialize with SOS token
    igpt_output = image_gpt.generate(
        input_ids=context,
        max_length=16 * 16 + 2,
        temperature=temperature,
        do_sample=True,
        top_k=45,
        top_p=0.9,
    )
    igpt_output = igpt_output[:, 1:]
    igpt_output[igpt_output == sos_token] = 0

    quantized = rearrange(embedding(igpt_output), "b t c -> t b c")
    features = quantized + decoder.pos_embedding

    features = rearrange(features, "t b c -> b t c")
    features = decoder.transformer(features)
    features = rearrange(features, "b t c -> t b c")
    features = features[1:]  # remove global feature

    patches = decoder.head(features)
    x_recon = decoder.patch2img(patches)

    return x_recon, igpt_output, torch.tensor([-1] * num_images)


@torch.no_grad()
def bootstrap_past_samples(
    image_gpt: ImageGPTForCausalImageModeling,
    vq_vae_model,
    num_images: int,
    experience_step: int,
    dataset_path: str,
    config: TrainConfig,
    sos_token: int,
    mask_token: int,
    transform: t.Optional[t.Any] = None,
    time_indices=None,
) -> ClassificationDataset:
    num_images_per_batch = min(128, num_images)

    bootstrapped_dataset = BootstrappedDataset(
        dataset_path=dataset_path,
        experience_step=experience_step,
        transform=transform,
    )
    image_embeddings = get_image_embedding(vq_vae_model, config, mask_token).to(
        vq_vae_model.device
    )

    for i in range(num_images // num_images_per_batch):
        images, latent_indices, sampled_time_indices = sample_images(
            image_gpt=image_gpt,
            vq_vae_model=vq_vae_model,
            embedding=image_embeddings,
            sos_token=sos_token,
            temperature=config.temperature,
            num_images=num_images_per_batch,
        )

        bootstrapped_dataset.add_data(
            images=images.cpu(),
            latent_indices=latent_indices.cpu(),
            time_indices=sampled_time_indices.cpu(),
        )

    dataset = make_classification_dataset(
        bootstrapped_dataset, targets=bootstrapped_dataset.targets
    )

    return dataset


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
    sos_token = config.num_class_embeddings + config.num_embeddings + 1
    mask_token = config.num_class_embeddings + config.num_embeddings

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
    if task_id is None:
        """
        If task_id is empty we calculate fid score for all tasks
        observed before given experience step
        """
        bootstrapped_dataset = bootstrap_past_samples(
            image_gpt=image_gpt,
            vq_vae_model=model,
            num_images=num_images,
            dataset_path=config.bootstrapped_dataset_path,
            config=config,
            sos_token=sos_token,
            experience_step=exp_step,
            mask_token=mask_token,
        )
        bootstrapped_dataset = ImgDataset(bootstrapped_dataset)

        real_dataset = ConcatDataset(
            [
                wrap_dataset_with_empty_indices(
                    experience.dataset, time_index=experience_step
                )
                for experience_step, experience in enumerate(
                    benchmark.train_stream[: exp_step + 1]
                )
            ]
        )
        real_dataset = ImgDataset(real_dataset)
    else:
        """
        If task_id is not None we calculate fid score for this task
        at given experience step.
        """
        time_indices = torch.tensor([task_id] * num_images).to(model.device)
        bootstrapped_dataset = bootstrap_past_samples(
            image_gpt=image_gpt,
            vq_vae_model=model,
            num_images=num_images,
            dataset_path=config.bootstrapped_dataset_path,
            config=config,
            sos_token=sos_token,
            experience_step=exp_step,
            mask_token=mask_token,
            time_indices=time_indices,
        )
        bootstrapped_dataset = ImgDataset(bootstrapped_dataset)

        real_dataset = wrap_dataset_with_empty_indices(
            benchmark.train_stream[task_id].dataset, time_index=exp_step
        )
        real_dataset = ImgDataset(real_dataset)

    return calculate_fid_given_datasets(
        bootstrapped_dataset, real_dataset, 128, device, 2048
    )


def calculate_fid_score_for_all_cl_steps(run_id, num_images, max_epochs, min_epochs):
    ini_config = ConfigParser()
    ini_config.read("./src/transformer_vq_vae/configuration/train.ini")

    config = TrainConfig.construct_typed_config(ini_config)

    run = wandb.init(
        id=run_id,
        project="transformer-vq-vae",
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

    vocab_size = config.num_class_embeddings + config.num_embeddings + 2
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
            "n_positions": 16 * 16 + 2,
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
    m_eps = [
        max_epochs,
        min_epochs,
        min_epochs,
        min_epochs,
        min_epochs,
    ]

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
                i_ep=9,
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

    for task_id in trange(len(benchmark.train_stream)):
        print(f"Compute score per task: {task_id}")

        task_fid_scores = []
        for experience_step in trange(
            task_id, len(benchmark.train_stream), leave=False
        ):
            task_fid_scores.append(
                calculate_fid_score(
                    config=config,
                    benchmark=benchmark,
                    model=model,
                    image_gpt=image_gpt,
                    device=device,
                    run_id=run_id,
                    exp_step=experience_step,
                    task_id=task_id,
                    m_ep=m_eps[experience_step],
                    i_ep=9,
                    num_images=num_images,
                )
            )

        # log to wandb
        wandb.log(
            {
                f"fid_score/experience_step_{task_id}": wandb.Table(
                    columns=["experience_step", "value"],
                    data=list(
                        zip(
                            range(task_id, len(benchmark.train_stream)), task_fid_scores
                        )
                    ),
                )
            }
        )

    wandb.finish()
