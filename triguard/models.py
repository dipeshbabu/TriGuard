import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    ViT_B_16_Weights,
    convnext_tiny,
    densenet121,
    resnet50,
    swin_t,
    vit_b_16,
)


IMAGENET_MODEL_SUFFIX = "_imagenet"


def split_model_name(name: str) -> tuple[str, bool]:
    name = name.lower()
    if name.endswith(IMAGENET_MODEL_SUFFIX):
        return name[: -len(IMAGENET_MODEL_SUFFIX)], True
    return name, False


def uses_imagenet_preprocessing(name: str) -> bool:
    _, pretrained = split_model_name(name)
    return pretrained


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
    def __init__(self, num_classes: int, grayscale: bool = False, weights=None):
        super().__init__()
        self.to_rgb = GrayscaleToRGB() if grayscale else nn.Identity()
        self.resize = nn.Upsample(
            size=(224, 224), mode="bilinear", align_corners=False)
        self.backbone = vit_b_16(weights=weights)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.to_rgb(x)
        x = self.resize(x)
        return self.backbone(x)


def get_model(name: str, dataset: str, num_classes: int = 10):
    name, pretrained = split_model_name(name)
    dataset = dataset.lower()
    grayscale = dataset in ["mnist", "fashionmnist"]

    if pretrained and name == "simplecnn":
        raise ValueError("SimpleCNN does not have an ImageNet-pretrained variant.")

    if name == "simplecnn":
        input_channels = 1 if grayscale else 3
        input_size = 28 if grayscale else 32
        return SimpleCNN(
            input_channels=input_channels,
            input_size=input_size,
            dropout_p=0.25,
            num_classes=num_classes,
        )

    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if not pretrained:
            input_channels = 1 if grayscale else 3
            model.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            model.maxpool = nn.Identity()
        return model

    if name == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        if not pretrained:
            input_channels = 1 if grayscale else 3
            model.features.conv0 = nn.Conv2d(
                input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            model.features.pool0 = nn.Identity()
        return model

    if name in ["vit_b_16", "vit"]:
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        return VisionTransformerWrapper(
            num_classes=num_classes,
            grayscale=(grayscale and not pretrained),
            weights=weights,
        )

    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if name == "swin_t":
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    raise ValueError(f"Unknown model: {name}")
