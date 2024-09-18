import glob
from pathlib import Path

import torch
import torchvision
from torchvision.io import read_image
from datasets import load_dataset

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, config, run_path):
        self.images = []
        self.labels = []

        synthetic_dataset_path = Path(run_path)

        # read dataset to memory
        sample_images = sorted(glob.glob(f"{synthetic_dataset_path}/*.jpg"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        sample_labels = sorted(glob.glob(f"{synthetic_dataset_path}/*.pt"), key=lambda x: int(x.split("_")[-1].split(".")[0]))

        for batched_images, batched_labels in zip(sample_images, sample_labels):
            batched_images = read_image(batched_images)
            batched_labels = torch.load(batched_labels).tolist()

            self.images.extend(
                [
                    batched_images[:, i * config.image_size : (i + 1) * config.image_size].float() / 255
                    for i in range(batched_images.shape[-2] // config.image_size)
                ]
            )
            self.labels.extend(batched_labels)

        # transform dataset
        self.preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((config.image_size, config.image_size)),
                torchvision.transforms.RandomHorizontalFlip(),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        image = self.preprocess(image)

        return image, self.labels[item]


class InitialDataset(torch.utils.data.Dataset):
    """
    ## Dataset

    This loads the training dataset and resize it to the give image size.
    """

    def __init__(self, image_size: int, dataset_slug: str):
        """
        * `path` path to the folder containing the images
        * `image_size` size of the image
        """
        super().__init__()

        # Get the paths of all `jpg` files
        self.dataset = load_dataset(dataset_slug, split="train")

        # Transformation
        self.transform = torchvision.transforms.Compose(
            [
                # Resize the image
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.RandomHorizontalFlip(),
                # Convert to PyTorch tensor
                torchvision.transforms.ToTensor(),
            ]
        )

    def __len__(self):
        """Number of images"""
        return len(self.dataset)

    def __getitem__(self, index):
        """Get the the `index`-th image"""
        label = self.dataset[index]['label']
        data = self.dataset[index]['image']

        # Ensure the image has 3 channels (RGB)
        if data.mode != 'RGB':
            data = data.convert('RGB')
        
        return self.transform(data), label


@torch.no_grad()
def sample_synthetic_dataset(configs, run_path):
    synthetic_dataset_path = Path(run_path)
    synthetic_dataset_path.mkdir(exist_ok=True, parents=True)

    for i in range(configs.synth_dataset_num_images // configs.synth_dataset_batch_size + 1):
        images, _, labels = configs.generate_images(configs.synth_dataset_batch_size)
        
        images = images.cpu()
        labels = labels.cpu()

        torchvision.utils.save_image(images, fp=f"{synthetic_dataset_path}/batch_{i}.jpg", nrow=1, padding=0)
        torch.save(labels, f"{synthetic_dataset_path}/batch_labels_{i}.pt")