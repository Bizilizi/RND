import argparse
import logging
from functools import partial

import torch
from lightning_fabric import seed_everything
from torch import distributed

import wandb
from src.rnd.train import main as rnd_main
from src.utils.train_script import parse_arguments
from src.vae_ft.train import main as vae_ft_main
from src.vq_vae.train import main as vq_vae_main
from src.transformer_vq_vae.train import main as transformer_vq_vae_main
from train_utils import add_arguments

# configure logging at the root level of Lightning
# logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
torch.multiprocessing.set_sharing_strategy("file_system")
# torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":

    # Parse arguments from command line
    parser = argparse.ArgumentParser(description="Model trainer")
    parser = add_arguments(parser)
    args = parse_arguments(parser)

    # Choose appropriate entry function depending on model name
    if args.model == "rnd":
        entry_main = rnd_main
    elif args.model == "vae-ft":
        entry_main = vae_ft_main
    elif args.model == "vq-vae":
        entry_main = vq_vae_main
    elif args.model == "transformer-vq-vae":
        entry_main = transformer_vq_vae_main
    else:
        assert False, "Unknown value '--model' parameter"

    # Make it deterministic
    seed_everything(args.seed)

    # Run wandb agent if sweep id was passed to arguments
    if args.sweep_id:
        wandb.agent(
            args.sweep_id,
            partial(entry_main, args),
            project="transformer-vq-vae",
        )
    else:
        # Otherwise just run entry main function
        entry_main(args)
