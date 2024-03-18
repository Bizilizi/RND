import pathlib
from configparser import ConfigParser

import wandb
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm.auto import trange
from src.transformer_vq_vae.configuration.config import TrainConfig
from src.transformer_vq_vae.init_scrips import get_benchmark, get_model
from src.transformer_vq_vae.utils.wrap_empty_indices import (
    wrap_dataset_with_empty_indices,
)
from train_utils import get_device
from torch_cka import CKA


class ImgDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, *_ = self.dataset[item]
        return x["images"], 1


def calculate_cka_score(
    *,
    config,
    benchmark,
    model_1,
    model_2,
    exp_step_1,
    exp_step_2,
    task_id,
    device,
    run_id,
    m_ep_1,
    m_ep_2,
    batch_size=512,
):
    ## WARNING: BER CAREFUL WITH CALCULATION OF THESE
    ## HAVE TO BE CONSISTENT WITH TRAINING SCRIPT

    model_1.load_state_dict(
        torch.load(
            f"/scratch/shared/beegfs/dzverev/artifacts/{run_id}/model/model-exp-{exp_step_1}-ep-{m_ep_1}.ckpt",
            map_location=torch.device("cpu"),
        )["state_dict"]
    )
    model_2.load_state_dict(
        torch.load(
            f"/scratch/shared/beegfs/dzverev/artifacts/{run_id}/model/model-exp-{exp_step_2}-ep-{m_ep_2}.ckpt",
            map_location=torch.device("cpu"),
        )["state_dict"]
    )

    model_1.to(device)
    model_2.to(device)

    # create datasets
    real_dataset = wrap_dataset_with_empty_indices(
        benchmark.train_stream[task_id].dataset, time_index=task_id
    )
    real_dataset = ImgDataset(real_dataset)
    dataloader = DataLoader(
        real_dataset,
        batch_size=batch_size,  # according to your device memory
        shuffle=False,
    )  # Don't forget to seed your dataloader

    cka = CKA(
        model_1,
        model_2,
        model1_name="model1",  # good idea to provide names to avoid confusion
        model2_name="model2",
        model1_layers=["encoder.layer_norm"],  # List of layers to extract features from
        model2_layers=["encoder.layer_norm"],  # extracts all layer features by default
        device="cuda",
    )

    cka.compare(dataloader)  # secondary dataloader is optional

    return cka.export()


def calculate_fid_score_for_all_cl_steps(run_id, batch_size):
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
    model_1 = get_model(config, device)
    model_2 = get_model(config, device)

    for task_id in trange(len(benchmark.train_stream)):
        print(f"Compute score per task: {task_id}")

        task_fid_scores = []
        for experience_step in trange(
            task_id, len(benchmark.train_stream), leave=False
        ):
            task_fid_scores.append(
                calculate_cka_score(
                    config=config,
                    benchmark=benchmark,
                    model_1=model_1,
                    model_2=model_2,
                    exp_step_1=task_id,
                    exp_step_2=experience_step,
                    task_id=task_id,
                    device=device,
                    run_id=run_id,
                    m_ep_1=900,
                    m_ep_2=300,
                    batch_size=batch_size,
                )["CKA"][0]
                .cpu()
                .item()
            )

        # log to wandb
        wandb.log(
            {
                f"cka_score/experience_step_{task_id}": wandb.Table(
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
