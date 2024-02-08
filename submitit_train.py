import argparse
from functools import partial
from subprocess import Popen

import submitit
import wandb
from lightning_fabric import seed_everything

from train_utils import add_arguments as add_train_arguments
from src.rnd.train import main as rnd_main
from src.utils.train_script import parse_arguments
from src.vae_ft.train import main as vae_ft_main
from src.vq_vae.train import main as vq_vae_main
from src.transformer_vq_vae.train import main as transformer_vq_vae_main


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def add_sumbitit_arguments(parser):
    parser.add_argument(
        "--ngpus", default=1, type=int, help="Number of gpus to request on each node"
    )
    parser.add_argument(
        "--timeout", default=60 * 48, type=int, help="Duration of the job"
    )
    parser.add_argument(
        "--partition", default="gpu", type=str, help="Partition where to submit"
    )
    parser.add_argument(
        "--gpu_type",
        default="p40",
        type=str,
    )
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes to request"
    )
    parser.add_argument(
        "--output_dir",
        default="./slurm_logs",
        type=str,
        help="Where stdout and stderr will write to",
    )
    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message",
    )
    parser.add_argument(
        "--slurm_mem",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--cpus_per_task",
        default=4,
        type=int,
    )

    return parser


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import os

        self._setup_gpu_args()
        os.environ["TIMM_FUSED_ATTN"] = "1"

        # Choose appropriate entry function depending on model name
        if self.args.model == "rnd":
            entry_main = rnd_main
        elif self.args.model == "vae-ft":
            entry_main = vae_ft_main
        elif self.args.model == "vq-vae":
            entry_main = vq_vae_main
        elif self.args.model == "transformer-vq-vae":
            entry_main = transformer_vq_vae_main
        else:
            assert False, "Unknown value '--model' parameter"

        # Make it deterministic
        seed_everything(self.args.seed)

        # Run wandb agent if sweep id was passed to arguments
        if self.args.sweep_id:
            wandb.agent(
                self.args.sweep_id,
                partial(entry_main, self.args),
                project="transformer-vq-vae",
            )
        else:
            # Otherwise just run entry main function
            entry_main(self.args)

    def checkpoint(self):
        import submitit
        import pathlib

        meta_file_path = (
            pathlib.Path(self.args.checkpoint_path) / "checkpoint_metadata.ckpt"
        )
        if meta_file_path.exists():
            self.args.resume_from = str(meta_file_path)

        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()

        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser(description="Model trainer")

    parser = add_train_arguments(parser)
    parser = add_sumbitit_arguments(parser)

    args = parse_arguments(parser)

    # Make it deterministic
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.comment:
        kwargs["slurm_comment"] = args.comment

    executor.update_parameters(
        mem_gb=args.slurm_mem,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=args.cpus_per_task,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_constraint=args.gpu_type,
        slurm_additional_parameters={
            "mail-type": "BEGIN,END,FAIL",
            "mail-user": "dzverev@robots.ox.ac.uk",
        },
        **kwargs,
    )
    executor.update_parameters(name=args.comment)

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
