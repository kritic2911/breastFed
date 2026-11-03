"""Model architecture and backbone definitions for mammography classification.

Adheres strictly to AGENTS.md Section 2.1:
  - ResNet-18 (ImageNet weights).
  - Grayscale input replicated to 3 channels.
  - Linear(512, 3) classification head.
  - Frozen: conv1, bn1, layer1, layer2.
  - Fine-tuned: layer3, layer4, and head.
  - BatchNorm mode: "frozen" vs "trainable" (decided in Phase P2 sweep).
  - Head learning rate is 10x the backbone learning rate.
"""

from typing import Dict, List, Literal

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class MammographyResNet18(nn.Module):
    """ResNet-18 adapted for 3-class mammography classification."""

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        bn_mode: Literal["frozen", "trainable"] = "frozen",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bn_mode = bn_mode

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        # Base components
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        # 3-class classification head Linear(512, 3)
        self.fc = nn.Linear(512, num_classes)

        # Freeze early feature extraction stages: conv1, bn1, layer1, layer2
        self._freeze_early_layers()

        # Configure BatchNorm behavior in layer3 and layer4
        self._configure_bn()

    def _freeze_early_layers(self):
        """Freeze conv1, bn1, layer1, and layer2 parameters."""
        for module in [self.conv1, self.bn1, self.layer1, self.layer2]:
            for param in module.parameters():
                param.requires_grad = False
            # Early BN layers are permanently in eval mode (ImageNet statistics)
            module.eval()

    def _configure_bn(self):
        """Configure BatchNorm layers in fine-tuned stages (layer3, layer4)."""
        if self.bn_mode == "frozen":
            for module in [self.layer3, self.layer4]:
                for m in module.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m.eval()
                        for p in m.parameters():
                            p.requires_grad = False
        elif self.bn_mode == "trainable":
            for module in [self.layer3, self.layer4]:
                for m in module.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m.train()
                        for p in m.parameters():
                            p.requires_grad = True
        else:
            raise ValueError(f"Unknown bn_mode: {self.bn_mode}. Expected 'frozen' or 'trainable'.")

    def train(self, mode: bool = True):
        """Override train to keep frozen layers and frozen BN in eval mode."""
        super().train(mode)
        # Always keep conv1, bn1, layer1, layer2 in eval mode
        self.conv1.eval()
        self.bn1.eval()
        self.layer1.eval()
        self.layer2.eval()

        # If bn_mode is frozen, keep layer3 and layer4 BN in eval mode
        if self.bn_mode == "frozen":
            for module in [self.layer3, self.layer4]:
                for m in module.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input:
            x: Tensor of shape (B, 1, 512, 512) or (B, 3, 512, 512).
        Returns:
            logits: Tensor of shape (B, num_classes).
        """
        # Replicate single grayscale channel to 3 channels for ImageNet backbone
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)
        return logits

    def get_parameter_groups(self, backbone_lr: float, head_lr_multiplier: float = 10.0) -> List[Dict[str, any]]:
        """Get optimizer parameter groups with 10x learning rate for the head."""
        head_params = list(self.fc.parameters())
        backbone_params = [
            p for p in list(self.layer3.parameters()) + list(self.layer4.parameters())
            if p.requires_grad
        ]
        return [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params, "lr": backbone_lr * head_lr_multiplier, "name": "head"},
        ]


def build_model(
    num_classes: int = 3,
    pretrained: bool = True,
    bn_mode: Literal["frozen", "trainable"] = "frozen",
) -> MammographyResNet18:
    """Build MammographyResNet18 model."""
    return MammographyResNet18(num_classes=num_classes, pretrained=pretrained, bn_mode=bn_mode)

