"""Model factory — ResNet18 with frozen BN for cross-scanner robustness."""

import torch
import torch.nn as nn
from torchvision import models


def build_model(cfg: dict, device: str = "cpu") -> nn.Module:
    """Build ResNet18 with ImageNet pretrained weights.

    - Replaces the final FC layer for binary classification (1 output → sigmoid).
    - If cfg["freeze_bn"] is True, all BatchNorm layers are frozen:
      their running_mean / running_var stay at ImageNet values,
      and weight/bias are not updated. This prevents cross-scanner
      batch-norm contamination in federated averaging.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg["pretrained"] else None
    model = models.resnet18(weights=weights)

    # Replace final FC: 512 → 1 (binary, trained with BCEWithLogitsLoss)
    model.fc = nn.Linear(model.fc.in_features, cfg["num_classes"])
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)

    if cfg.get("freeze_bn", True):
        freeze_batchnorm(model)

    return model.to(device)


def freeze_batchnorm(model: nn.Module):
    """Freeze all BatchNorm layers: set to eval mode, disable gradient."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def get_trainable_params(model: nn.Module):
    """Return only parameters that require grad (excludes frozen BN)."""
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def model_size_bytes(model: nn.Module) -> int:
    """Estimate model size in bytes (for communication cost tracking)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def get_bn_keys(model: nn.Module) -> list:
    """Return state_dict keys that belong to BatchNorm layers.

    Used by FedBN to exclude BN params from averaging.
    """
    bn_keys = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for pname in ["weight", "bias", "running_mean", "running_var",
                          "num_batches_tracked"]:
                full_key = f"{name}.{pname}"
                bn_keys.append(full_key)
    return bn_keys
