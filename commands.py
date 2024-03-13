import argparse

from src.transformer_vq_vae.commands.compute_fid_score import (
    calculate_fid_score_for_all_cl_steps,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model trainer")
    parser.add_argument(
        "--run_id",
        type=str,
        help="Id of wandb run",
    )
    parser.add_argument(
        "--command",
        type=str,
        help="name of the command",
    )

    args = parser.parse_args()

    if args.command == "fid_score":
        calculate_fid_score_for_all_cl_steps(args.run_id)
