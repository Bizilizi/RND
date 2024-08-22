import argparse

from .main import Configs, restore_from_previous_step, sample_synthetic_dataset


def main(restore_from):
    assert restore_from, "argument restore_from cannot be None or empty"

    # Create configurations object
    configs = Configs()
    _, synthetic_dataset_path = restore_from_previous_step(configs, restore_from)
    sample_synthetic_dataset(configs, synthetic_dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stylegan trainer")
    parser.add_argument("--restore_from", type=str, help="experiment path", default=None)
    args = parser.parse_args()

    main(args.restore_from)
