from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
from torchvision import transforms

# CIFAR10
cifar_augmentations = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
)
cifar10_to_tensor_and_normalization = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
    ]
)
default_cifar10_train_transform = transforms.Compose(
    [cifar_augmentations, cifar10_to_tensor_and_normalization]
)
default_cifar10_eval_transform = cifar10_to_tensor_and_normalization

# CIFAR100
cifar100_to_tensor_and_normalization = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (1, 1, 1)),
    ]
)
default_cifar100_train_transform = transforms.Compose(
    [
        cifar_augmentations,
        cifar100_to_tensor_and_normalization,
    ]
)

default_cifar100_eval_transform = cifar100_to_tensor_and_normalization

# ImageNet
# MAE recipe https://github.com/facebookresearch/mae/blob/main/util/datasets.py
imagenet_to_tensor_and_normalization = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
imagenet_augmentations = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
    ]
)
default_imagenet_train_transform = transforms.Compose(
    [
        imagenet_augmentations,
        imagenet_to_tensor_and_normalization,
    ]
)

default_imagenet_eval_transform = imagenet_to_tensor_and_normalization
