import argparse
import joblib
import wandb

import train

from copy import deepcopy
import submitit


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
        "--jobs_per_task",
        default=8,
        type=int,
    )

    train.add_arguments(parser)

    return parser.parse_args()


class Trainer(object):
    def __init__(self, m_args):
        self.m_args = m_args

    def __call__(self):
        import train

        wandb.setup()
        joblib.Parallel(n_jobs=len(self.m_args))(
            joblib.delayed(train.main)(args) for args in self.m_args
        )

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
        mem_gb=12 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=8,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
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

    all_arguments = []
    for num_random_images in [0, 1_000, 3_000, 5_000, 7_000, 10_000]:
        for num_random_noise in [100, 500, 1_000, 3_000, 5_000]:
            args_new = deepcopy(args)

            args_new.config = "src/vae_ft/configuration/train.ini"
            args_new.model = "vae-ft"
            args_new.train_logger = "wandb"
            args_new.evaluation_logger = "wandb"
            args_new.model_backbone = "mlp"
            args_new.max_epochs = 100
            args_new.group = "r_images vs r_noise"
            args_new.num_workers = 0

            args_new.num_random_images = num_random_images
            args_new.num_random_noise = num_random_noise
            args_new.experiment_name = (
                f"MLP.RI-{num_random_images}.RN-{num_random_noise}"
            )

            all_arguments.append(args_new)

    args_per_gpu = chunker(all_arguments, args.jobs_per_task)
    all_trainers = [Trainer(m_args) for m_args in args_per_gpu]

    jobs = executor.submit_array(all_trainers)

    for i, (j, t) in enumerate(zip(jobs, all_trainers)):
        print(f"Submitted job_id: {j.job_id}")


if __name__ == "__main__":
    main()
