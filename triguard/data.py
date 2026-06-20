import os

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _worker_count(requested: int | None = None) -> int:
    if requested is not None:
        return max(0, requested)
    cpu = os.cpu_count() or 0
    return min(4, cpu)


def _imagenet_transform(train: bool, grayscale: bool):
    steps = []
    if grayscale:
        steps.append(T.Grayscale(num_output_channels=3))
    if train:
        steps.extend(
            [
                T.Resize(256),
                T.RandomCrop(224, padding=4),
                T.RandomHorizontalFlip(),
            ]
        )
    else:
        steps.extend([T.Resize(256), T.CenterCrop(224)])
    steps.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose(steps)


def _native_transform(train: bool, mean=None, std=None):
    # Keep the original workshop preprocessing unchanged. Stronger augmentation
    # is only used by the ImageNet-pretrained profile.
    steps = []
    steps.append(T.ToTensor())
    if mean is not None and std is not None:
        steps.append(T.Normalize(mean, std))
    return T.Compose(steps)


def _norm_bounds(mean, std):
    clamp_min = min((0.0 - m) / s for m, s in zip(mean, std))
    clamp_max = max((1.0 - m) / s for m, s in zip(mean, std))
    baseline_min, baseline_max = clamp_min, clamp_max
    return clamp_min, clamp_max, baseline_min, baseline_max


def _norm_eps(pixel_eps: float, std):
    return tuple(pixel_eps / s for s in std)


def get_dataset(
    name: str,
    data_root: str = "./data",
    input_profile: str = "native",
):
    name = name.lower()
    input_profile = input_profile.lower()
    if input_profile not in {"native", "imagenet"}:
        raise ValueError(f"Unknown input profile: {input_profile}")
    imagenet_profile = input_profile == "imagenet"

    if name == "mnist":
        if imagenet_profile:
            train_tfm = _imagenet_transform(train=True, grayscale=True)
            test_tfm = _imagenet_transform(train=False, grayscale=True)
            clamp_min, clamp_max, baseline_min, baseline_max = _norm_bounds(
                IMAGENET_MEAN, IMAGENET_STD
            )
            eps = _norm_eps(0.3, IMAGENET_STD)
        else:
            train_tfm = test_tfm = T.Compose([T.ToTensor()])
            clamp_min, clamp_max = 0.0, 1.0
            eps = 0.3
            baseline_min, baseline_max = 0.0, 1.0
        train = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=train_tfm)
        test = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=test_tfm)
        num_classes = 10
        pixel_eps = 0.3

    elif name == "fashionmnist":
        if imagenet_profile:
            train_tfm = _imagenet_transform(train=True, grayscale=True)
            test_tfm = _imagenet_transform(train=False, grayscale=True)
            clamp_min, clamp_max, baseline_min, baseline_max = _norm_bounds(
                IMAGENET_MEAN, IMAGENET_STD
            )
            eps = _norm_eps(0.3, IMAGENET_STD)
        else:
            train_tfm = test_tfm = T.Compose([T.ToTensor()])
            clamp_min, clamp_max = 0.0, 1.0
            eps = 0.3
            baseline_min, baseline_max = 0.0, 1.0
        train = torchvision.datasets.FashionMNIST(data_root, train=True, download=True, transform=train_tfm)
        test = torchvision.datasets.FashionMNIST(data_root, train=False, download=True, transform=test_tfm)
        num_classes = 10
        pixel_eps = 0.3

    elif name == "cifar10":
        if imagenet_profile:
            train_tfm = _imagenet_transform(train=True, grayscale=False)
            test_tfm = _imagenet_transform(train=False, grayscale=False)
            clamp_min, clamp_max, baseline_min, baseline_max = _norm_bounds(
                IMAGENET_MEAN, IMAGENET_STD
            )
            eps = _norm_eps(8 / 255, IMAGENET_STD)
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
            train_tfm = _native_transform(train=True, mean=mean, std=std)
            test_tfm = _native_transform(train=False, mean=mean, std=std)
            clamp_min, clamp_max = -1.0, 1.0
            eps = (8 / 255) / 0.5
            baseline_min, baseline_max = -1.0, 1.0
        train = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=train_tfm)
        test = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=test_tfm)
        num_classes = 10
        pixel_eps = 8 / 255

    elif name == "cifar100":
        if imagenet_profile:
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
            train_tfm = _imagenet_transform(train=True, grayscale=False)
            test_tfm = _imagenet_transform(train=False, grayscale=False)
        else:
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
            train_tfm = _native_transform(train=True, mean=mean, std=std)
            test_tfm = _native_transform(train=False, mean=mean, std=std)
        train = torchvision.datasets.CIFAR100(data_root, train=True, download=True, transform=train_tfm)
        test = torchvision.datasets.CIFAR100(data_root, train=False, download=True, transform=test_tfm)
        clamp_min, clamp_max, baseline_min, baseline_max = _norm_bounds(mean, std)
        eps = _norm_eps(8 / 255, std)
        num_classes = 100
        pixel_eps = 8 / 255

    else:
        raise ValueError(f"Unknown dataset: {name}")

    meta = {
        "num_classes": num_classes,
        "baseline_min": baseline_min,
        "baseline_max": baseline_max,
        "input_profile": input_profile,
        "pixel_eps": pixel_eps,
    }
    return train, test, clamp_min, clamp_max, eps, meta


def get_loaders(
    name: str,
    batch_size: int,
    test_batch: int = 256,
    num_workers: int | None = None,
    data_root: str = "./data",
    input_profile: str = "native",
):
    train, test, clamp_min, clamp_max, eps, meta = get_dataset(
        name,
        data_root=data_root,
        input_profile=input_profile,
    )
    workers = _worker_count(num_workers)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
    )
    test_loader = DataLoader(
        test,
        batch_size=test_batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
    )

    return train_loader, test_loader, test, clamp_min, clamp_max, eps, meta
