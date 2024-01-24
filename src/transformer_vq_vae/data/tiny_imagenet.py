import os
import typing as t
from pathlib import Path

from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from avalanche.benchmarks import NCScenario, nc_benchmark

_default_cifar10_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

_default_cifar10_eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


class TinyImageNet(VisionDataset):
    """TinyImageNet <http://vision.stanford.edu/teaching/cs231n/reports/2015/pdfs/yle_project.pdf>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``tiny-imagenet-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "tiny-imagenet-200"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    tgz_md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    train_list = ["train"]

    test_list = ["test"]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: t.Optional[t.Callable] = None,
        target_transform: t.Optional[t.Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.root = root

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        self.data: t.Any = []
        self.targets = []

        # load classes names
        classes_file = os.path.join(self.root, self.base_folder, "wnids.txt")
        classes_dict = {}
        with open(classes_file, "r") as f:
            lines = f.readlines()
            for class_name, class_id in zip(lines, range(len(lines))):
                class_name = class_name.strip()
                classes_dict[class_name] = class_id

        # now load images paths
        if train:
            self._load_train(classes_dict)
        else:
            self._load_val(classes_dict)

    def _load_train(self, classes_dict):
        file_name = "train"
        classes_dir_path = os.path.join(self.root, self.base_folder, file_name)

        for class_dir in os.listdir(classes_dir_path):
            images_path = os.path.join(classes_dir_path, class_dir, "images")
            for image_name in os.listdir(images_path):
                self.data.append(os.path.join(images_path, image_name))
                self.targets.append(classes_dict[class_dir])

    def _load_val(self, classes_dict):
        val_classes_file = os.path.join(
            self.root, self.base_folder, "val", "val_annotations.txt"
        )
        image_name_to_class = {}

        # build dict to map image name to class
        with open(val_classes_file, "r") as f:
            for line in f.readlines():
                (image_name, class_name, *_) = line.strip().split("\t")
                image_name_to_class[image_name] = classes_dict[class_name]

        # read image path
        images_paths = os.path.join(self.root, self.base_folder, "val", "images")
        for image_name in os.listdir(images_paths):
            self.data.append(os.path.join(images_paths, image_name))
            self.targets.append(image_name_to_class[image_name])

    def __getitem__(self, index: int) -> t.Tuple[t.Any, t.Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not os.path.exists(fpath):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


def SplitTinyImageNet(
    n_experiences: int,
    *,
    first_exp_with_half_classes: bool = False,
    return_task_id=False,
    seed: t.Optional[int] = None,
    fixed_class_order: t.Optional[t.Sequence[int]] = None,
    shuffle: bool = True,
    class_ids_from_zero_in_each_exp: bool = False,
    train_transform: t.Optional[t.Any] = _default_cifar10_train_transform,
    eval_transform: t.Optional[t.Any] = _default_cifar10_eval_transform,
    dataset_root: t.Union[str, Path] = None,
) -> NCScenario:
    """ """
    dataset_root = dataset_root or "./"
    cifar_train = TinyImageNet(dataset_root, train=True, download=True)
    cifar_test = TinyImageNet(dataset_root, train=False, download=True)

    return nc_benchmark(
        train_dataset=cifar_train,
        test_dataset=cifar_test,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        per_exp_classes={0: 5} if first_exp_with_half_classes else None,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
