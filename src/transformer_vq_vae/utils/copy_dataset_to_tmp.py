import os
import pathlib
import shutil

from src.transformer_vq_vae.configuration.config import TrainConfig


def copy_dataset_to_tmp(config: TrainConfig, target_dataset_dir):
    datasets_dir = pathlib.Path(config.dataset_path)

    if config.dataset == "cifar10":
        zip_path = datasets_dir / "cifar-10-python.tar.gz"
        dataset_path = datasets_dir / "cifar-10-batches-py"

        target_zip_path = target_dataset_dir / "cifar-10-python.tar.gz"
        target_dataset_path = target_dataset_dir / "cifar-10-batches-py"

        if zip_path.exists() and not target_zip_path.exists():
            shutil.copy(str(zip_path), str(target_zip_path))

        if dataset_path.exists() and not target_dataset_path.exists():
            shutil.copytree(str(dataset_path), str(target_dataset_path))

    if config.dataset == "cifar100":
        zip_path = datasets_dir / "cifar-100-python.tar.gz"
        target_zip_path = target_dataset_dir / "cifar-10-python.tar.gz"

        if zip_path.exists() and not target_zip_path.exists():
            shutil.copy(str(zip_path), str(target_zip_path))

    elif config.dataset == "tiny-imagenet":
        zip_path = datasets_dir / "tiny-imagenet-200.zip"
        target_zip_path = target_dataset_dir / "tiny-imagenet-200.zip"

        if zip_path.exists() and not target_zip_path.exists():
            shutil.copy(str(zip_path), str(target_zip_path))

    elif config.dataset == "imagenet":
        zip_path = datasets_dir / "ILSVRC2012_devkit_t12.tar.gz"
        target_zip_path = target_dataset_dir / "ILSVRC2012_devkit_t12.tar.gz"

        if zip_path.exists() and not target_zip_path.exists():
            shutil.copy(str(zip_path), str(target_zip_path))

        os.symlink(
            "/scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train",
            f"{target_dataset_dir}/train",
        )
        os.symlink(
            "/scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/val",
            f"{target_dataset_dir}/val",
        )
