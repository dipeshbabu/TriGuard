import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    densenet121,
    mobilenet_v3_large,
    resnet50,
    resnet101,
    vit_b_16,
)


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, input_size=28, dropout_p=0.25, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
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
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GrayscaleToRGB(nn.Module):
    def forward(self, x):
        if x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        return x


class VisionTransformerWrapper(nn.Module):
    def __init__(self, num_classes: int, grayscale: bool = False):
        super().__init__()
        self.to_rgb = GrayscaleToRGB() if grayscale else nn.Identity()
        self.resize = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
        self.backbone = vit_b_16(num_classes=num_classes)

    def forward(self, x):
        x = self.to_rgb(x)
        x = self.resize(x)
        return self.backbone(x)


def get_model(name: str, dataset: str, num_classes: int = 10):
    name = name.lower()
    dataset = dataset.lower()
    grayscale = dataset in ["mnist", "fashionmnist"]

    if name == "simplecnn":
        input_channels = 1 if grayscale else 3
        input_size = 28 if grayscale else 32
        return SimpleCNN(input_channels=input_channels, input_size=input_size, dropout_p=0.25, num_classes=num_classes)

    if name == "resnet50":
        model = resnet50(num_classes=num_classes)
        if grayscale:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model

    if name == "resnet101":
        model = resnet101(num_classes=num_classes)
        if grayscale:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model

    if name == "mobilenetv3":
        model = mobilenet_v3_large(num_classes=num_classes)
        if grayscale:
            model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        return model

    if name == "densenet121":
        model = densenet121(num_classes=num_classes)
        if grayscale:
            model.features.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.features.pool0 = nn.Identity()
        return model

    if name in ["vit_b_16", "vit"]:
        return VisionTransformerWrapper(num_classes=num_classes, grayscale=grayscale)

    raise ValueError(f"Unknown model: {name}")


def remove_dropout_layers(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            setattr(module, name, nn.Identity())
        else:
            remove_dropout_layers(child)
