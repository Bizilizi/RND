import typing as t
from pathlib import Path

from avalanche.benchmarks import NIScenario
from avalanche.benchmarks.datasets import default_dataset_location
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from src.avalanche.scenarios.ni_generator import ni_benchmark_rand_normal

_default_mnist_train_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

_default_mnist_eval_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


def _get_mnist_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location("mnist")

    train_set = MNIST(root=dataset_root, train=True, download=True)

    test_set = MNIST(root=dataset_root, train=False, download=True)

    return train_set, test_set


def NormalPermutedMNIST(
    *,
    sigma_range_constant: int = 4,
    train_transform: t.Optional[t.Any] = _default_mnist_train_transform,
    eval_transform: t.Optional[t.Any] = _default_mnist_eval_transform,
    dataset_root: t.Union[str, Path] = None
) -> NIScenario:
    """
    Creates a Permuted MNIST benchmark with use of ni_benchmark_rand_normal.

    It hase the same code as PermutedMNIST but uses different nc_generator.
    """

    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)

    return ni_benchmark_rand_normal(
        mnist_train,
        mnist_test,
        sigma_range_constant=sigma_range_constant,
        task_labels=False,
        shuffle=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
