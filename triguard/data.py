import os
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def _worker_count(requested: int | None = None) -> int:
    if requested is not None:
        return max(0, requested)
    cpu = os.cpu_count() or 0
    return min(4, cpu)


def get_dataset(name: str, data_root: str = "./data"):
    name = name.lower()

    if name == "mnist":
        tfm = T.Compose([T.ToTensor()])
        train = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=tfm)
        clamp_min, clamp_max = 0.0, 1.0
        eps = 0.3
        num_classes = 10
        baseline_min, baseline_max = 0.0, 1.0

    elif name == "fashionmnist":
        tfm = T.Compose([T.ToTensor()])
        train = torchvision.datasets.FashionMNIST(data_root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.FashionMNIST(data_root, train=False, download=True, transform=tfm)
        clamp_min, clamp_max = 0.0, 1.0
        eps = 0.3
        num_classes = 10
        baseline_min, baseline_max = 0.0, 1.0

    elif name == "cifar10":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=tfm)
        clamp_min, clamp_max = -1.0, 1.0
        eps = (8 / 255) / 0.5
        num_classes = 10
        baseline_min, baseline_max = -1.0, 1.0

    elif name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train = torchvision.datasets.CIFAR100(data_root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.CIFAR100(data_root, train=False, download=True, transform=tfm)
        clamp_min = min((0.0 - m) / s for m, s in zip(mean, std))
        clamp_max = max((1.0 - m) / s for m, s in zip(mean, std))
        eps = max((8 / 255) / s for s in std)
        num_classes = 100
        baseline_min, baseline_max = clamp_min, clamp_max

    else:
        raise ValueError(f"Unknown dataset: {name}")

    meta = {
        "num_classes": num_classes,
        "baseline_min": baseline_min,
        "baseline_max": baseline_max,
    }
    return train, test, clamp_min, clamp_max, eps, meta


def get_loaders(
    name: str,
    batch_size: int,
    test_batch: int = 256,
    num_workers: int | None = None,
    data_root: str = "./data",
):
    train, test, clamp_min, clamp_max, eps, meta = get_dataset(name, data_root=data_root)
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
