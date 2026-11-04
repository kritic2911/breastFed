"""High-resolution patch-level calcification expert with Multiple Instance Learning (MIL) pooling.

Adheres strictly to AGENTS.md Phase P3b, Rule 9 Geometry, and Decision D-010:
  - Input: Bag of K native-resolution 256x256 ROI patches extracted around detector clusters.
  - Instance backbone: Lightweight convolutional feature extractor (Conv2d, BatchNorm, ReLU, AdaptiveAvgPool).
  - Weak supervision: Patches inherit the image/breast-level diagnostic label (0=normal, 1=benign, 2=malignant).
  - Multiple Instance Learning (MIL) Aggregation:
      * Max pooling: Breast malignancy score = max(patch_scores)
      * Attention pooling: Gated attention weighting patches dynamically.
  - Zero-cluster handling: Empty bags receive a calibrated neutral baseline score.
  - Decision fusion: Logistic combination with global 512x512 ResNet-18 model.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchConvNet(nn.Module):
    """Lightweight 4-block ConvNet for 256x256 high-resolution microcalcification patches."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 256x256 -> 128x128
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 2: 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 3: 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 4: 32x32 -> 16x16
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

        x: (B, 1, 256, 256) tensor.
        Returns:
            logits: (B, 1) patch malignancy logit.
            embed: (B, embed_dim) patch embedding vector.
        """
        feats = self.features(x)
        embed = feats.flatten(1)
        logits = self.classifier(embed)
        return logits, embed


class CalcificationExpertMIL(nn.Module):
    """Multiple Instance Learning (MIL) model aggregating multiple ROI patches per breast."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
        pooling_mode: str = "max",
    ):
        super().__init__()
        self.pooling_mode = pooling_mode.lower()
        self.patch_net = PatchConvNet(in_channels=in_channels, embed_dim=embed_dim)

        # Gated Attention Module for attention pooling
        if self.pooling_mode == "attention":
            self.attention_a = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.Tanh(),
            )
            self.attention_b = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.Sigmoid(),
            )
            self.attention_w = nn.Linear(64, 1)
            self.bag_classifier = nn.Linear(embed_dim, 1)

        # Baseline logit for images with 0 detected clusters
        self.empty_bias = nn.Parameter(torch.tensor([-1.5], dtype=torch.float32))

    def forward_bag(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for a single image's bag of ROI patches.

        Args:
            patches: Tensor of shape (K, 1, 256, 256), where K is number of patches in bag.
                     If K == 0 (no clusters), returns empty_bias.

        Returns:
            bag_logit: Scalar tensor representing breast malignancy logit.
            patch_scores: (K,) tensor of individual patch malignancy probabilities.
        """
        if patches is None or patches.shape[0] == 0:
            return self.empty_bias, None

        patch_logits, embeds = self.patch_net(patches)  # (K, 1), (K, embed_dim)
        patch_probs = torch.sigmoid(patch_logits.squeeze(-1))

        if self.pooling_mode == "max":
            # MIL Max Pooling: breast is malignant if most suspicious patch is malignant
            bag_logit = torch.max(patch_logits)
        elif self.pooling_mode == "top_k":
            k_val = min(3, len(patch_logits))
            top_vals, _ = torch.topk(patch_logits.squeeze(-1), k=k_val)
            bag_logit = torch.mean(top_vals)
        elif self.pooling_mode == "attention":
            a = self.attention_a(embeds)
            b = self.attention_b(embeds)
            weights = self.attention_w(a * b)  # (K, 1)
            attn_norm = F.softmax(weights, dim=0)  # (K, 1)
            bag_embed = torch.sum(attn_norm * embeds, dim=0, keepdim=True)  # (1, embed_dim)
            bag_logit = self.bag_classifier(bag_embed).squeeze()
        else:
            bag_logit = torch.mean(patch_logits)

        return bag_logit, patch_probs

