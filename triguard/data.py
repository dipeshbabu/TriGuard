import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_dataset(name: str):
    name = name.lower()

    if name == "mnist":
        tfm = T.Compose([T.ToTensor()])
        train = torchvision.datasets.MNIST(
            "./data", train=True, download=True, transform=tfm)
        test = torchvision.datasets.MNIST(
            "./data", train=False, download=True, transform=tfm)
        clamp_min, clamp_max = 0.0, 1.0
        eps = 0.3  # paper protocol

    elif name == "fashionmnist":
        tfm = T.Compose([T.ToTensor()])
        train = torchvision.datasets.FashionMNIST(
            "./data", train=True, download=True, transform=tfm)
        test = torchvision.datasets.FashionMNIST(
            "./data", train=False, download=True, transform=tfm)
        clamp_min, clamp_max = 0.0, 1.0
        eps = 0.3  # paper protocol

    elif name == "cifar10":
        # IMPORTANT: 3-channel normalization
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train = torchvision.datasets.CIFAR10(
            "./data", train=True, download=True, transform=tfm)
        test = torchvision.datasets.CIFAR10(
            "./data", train=False, download=True, transform=tfm)

        # normalized input range corresponding to raw [0,1]
        clamp_min, clamp_max = -1.0, 1.0

        # eps in pixel space = 8/255, convert to normalized space by dividing by std=0.5
        eps = (8/255) / 0.5

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train, test, clamp_min, clamp_max, eps


def get_loaders(name: str, batch_size: int, test_batch: int = 256, num_workers: int = 2):
    train, test, clamp_min, clamp_max, eps = get_dataset(name)
    train_loader = DataLoader(train, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=test_batch,
                             shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, test, clamp_min, clamp_max, eps
