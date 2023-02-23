import argparse
import random
from subprocess import Popen

import joblib
import submitit

import train
import wandb


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ngpus", default=1, type=int, help="Number of gpus to request on each node"
    )
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes to request"
    )
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")
    parser.add_argument(
        "--partition", default="gpu", type=str, help="Partition where to submit"
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
        "--agents_per_task",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--num_tasks",
        default=3,
        type=int,
    )

    train.add_arguments(parser)

    return parser.parse_args()


class Trainer(object):
    def __init__(self, agents_per_task, args):
        self.agents_per_task = agents_per_task
        self.args = args

    def __call__(self):
        processes = [
            Popen(
                ["python", "train.py"]
                + [str(i) for k_v in self.args.items() for i in k_v]
            )
            for _ in range(self.agents_per_task)
        ]

        for p in processes:
            p.wait()

    def checkpoint(self):
        # TODO: Need to write this (used during pre-emption)
        ...


def main():
    args = parse_args()

    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.comment:
        kwargs["slurm_comment"] = args.comment

    executor.update_parameters(
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=20,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_mem=0,
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_constraint="p40",
        slurm_additional_parameters={
            "mail-type": "BEGIN,END,FAIL",
            "mail-user": "dzverev@robots.ox.ac.uk",
        },
        **kwargs,
    )

    executor.update_parameters(name="att_train")

    exp_args = {}

    exp_args["--config"] = "src/vae_ft/configuration/train.ini"
    exp_args["--model"] = "vae-ft"
    exp_args["--train_logger"] = "wandb"
    exp_args["--evaluation_logger"] = "wandb"
    exp_args["--max_epochs"] = 100
    exp_args["--num_workers"] = 4
    exp_args["--sweep_id"] = args.sweep_id

    all_trainers = [
        Trainer(args.agents_per_task, exp_args) for _ in range(args.num_tasks)
    ]

    jobs = executor.submit_array(all_trainers)

    for i, (j, t) in enumerate(zip(jobs, all_trainers)):
        print(f"Submitted job_id: {j.job_id}")


if __name__ == "__main__":
    main()
