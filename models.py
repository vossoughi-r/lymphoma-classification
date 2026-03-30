"""
Model architectures for lymphoma histopathology classification.

Two models:
    1. LymphomaNet  — custom lightweight CNN (trained from scratch)
    2. resnet18      — ImageNet-pretrained, fine-tuned on lymphoma data
"""

import torch
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────────────────────────
# Custom CNN
# ─────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv3x3 → BatchNorm → ReLU (×2)"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LymphomaNet(nn.Module):
    """
    Lightweight CNN for histopathology patches.

    Architecture:
        4 conv blocks (64→128→256→512) with progressive dropout
        → AdaptiveAvgPool → FC(512→256→num_classes)

    Args:
        num_classes: number of output classes (default: 3)
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),   nn.MaxPool2d(2), nn.Dropout2d(0.1),
            ConvBlock(64, 128), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            ConvBlock(128, 256), nn.MaxPool2d(2), nn.Dropout2d(0.3),
            ConvBlock(256, 512), nn.MaxPool2d(2), nn.Dropout2d(0.4),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────
# ResNet-18 transfer learning
# ─────────────────────────────────────────────────────────────

def get_resnet18(num_classes=3, freeze=True):
    """
    ResNet-18 with ImageNet weights, custom classification head.

    Args:
        num_classes: number of output classes
        freeze: if True, only layer4 + fc are trainable
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))

    if freeze:
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    return model


def unfreeze(model):
    """Unfreeze all layers for full fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True
