import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, resnet101, mobilenet_v3_large, densenet121


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, input_size=28, dropout_p=0.25, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_p)

        self.feature_dim = self._calculate_features(input_channels, input_size)
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _calculate_features(self, channels, size):
        dummy = torch.zeros(1, channels, size, size)
        dummy = self.pool(F.relu(self.conv1(dummy)))
        dummy = self.pool(F.relu(self.conv2(dummy)))
        return dummy.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(name: str, dataset: str, num_classes: int = 10):
    name = name.lower()
    dataset = dataset.lower()

    if name == "simplecnn":
        input_channels = 1 if dataset in ["mnist", "fashionmnist"] else 3
        input_size = 28 if dataset in ["mnist", "fashionmnist"] else 32
        return SimpleCNN(input_channels=input_channels, input_size=input_size, dropout_p=0.25, num_classes=num_classes)

    elif name == "resnet50":
        model = resnet50(num_classes=num_classes)
        if dataset in ["mnist", "fashionmnist"]:
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model

    elif name == "resnet101":
        model = resnet101(num_classes=num_classes)
        if dataset in ["mnist", "fashionmnist"]:
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model

    elif name == "mobilenetv3":
        model = mobilenet_v3_large(num_classes=num_classes)
        if dataset in ["mnist", "fashionmnist"]:
            model.features[0][0] = nn.Conv2d(
                1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        return model

    elif name == "densenet121":
        model = densenet121(num_classes=num_classes)
        if dataset in ["mnist", "fashionmnist"]:
            model.features.conv0 = nn.Conv2d(
                1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.features.pool0 = nn.Identity()
        return model

    else:
        raise ValueError(f"Unknown model: {name}")


def remove_dropout_layers(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            setattr(module, name, nn.Identity())
        else:
            remove_dropout_layers(child)
