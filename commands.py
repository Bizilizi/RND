import argparse

from src.transformer_vq_vae.commands.compute_cka_score import (
    calculate_cka_score_for_all_cl_steps,
)
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
    parser.add_argument(
        "--num_images", type=int, help="Number of images", default=25000
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=900)
    parser.add_argument("--min_epochs", type=int, default=300)
    args = parser.parse_args()

    if args.command == "fid_score":
        calculate_fid_score_for_all_cl_steps(args.run_id, args.num_images)
    elif args.command == "cka_score":
        calculate_cka_score_for_all_cl_steps(
            args.run_id,
            args.batch_size,
            args.max_epochs,
            args.min_epochs,
        )
