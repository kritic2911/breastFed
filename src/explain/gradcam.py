"""Grad-CAM computation, heat mass quantification, and anatomical cell attribution.

Adheres strictly to AGENTS.md Phase P3b:
  - Generates Grad-CAM localization maps from final convolutional layer (layer4).
  - Computes percentage of heat mass across the 3x3 anatomical grid cells.
  - Computes highlighted region area as a percentage of the segmented breast area.
  - Identifies peak activation coordinate (y*, x*) and maps to anatomical quadrant.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.explain.grid import BreastGrid, segment_breast_tissue


@dataclass
class LocalizationResult:
    """Quantitative results from Grad-CAM spatial localization."""
    class_idx: int
    target_logit: float
    target_prob: float
    heatmap: np.ndarray  # Shape: (512, 512), values in [0, 1]
    peak_coord: Tuple[int, int]  # (y*, x*)
    peak_cell_name: str  # e.g., "upper-posterior"
    highlighted_area_pixels: int
    highlighted_area_pct_breast: float  # e.g., 7.4%
    cell_heat_fractions: Dict[str, float]  # fraction of heat mass in each cell
    dominant_cell_name: str  # cell with highest integrated heat mass
    dominant_cell_fraction: float


class GradCAM:
    """Computes gradient-weighted class activation maps for ResNet-18 mammography models."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.model.eval()

        # Find target layer (default to layer4 of ResNet-18)
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
            self.target_layer = model.backbone.layer4[-1]
        elif hasattr(model, "layer4"):
            self.target_layer = model.layer4[-1]
        else:
            raise ValueError("Could not find layer4 in model architecture for Grad-CAM.")

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        img_tensor: torch.Tensor,
        class_idx: int = 2,
        label_family: str | None = None,
    ) -> Tuple[np.ndarray, float, float]:
        """Compute Grad-CAM heatmap for a single image tensor of shape (1, 3, 512, 512)."""
        self.model.zero_grad()

        # Forward pass
        if hasattr(self.model, "heads") and label_family is not None:
            logits = self.model(img_tensor, family=label_family)
        else:
            logits = self.model(img_tensor)

        n_classes = logits.shape[1]
        target_class = min(class_idx, n_classes - 1)
        target_logit = float(logits[0, target_class].item())
        target_prob = float(F.softmax(logits, dim=1)[0, target_class].item())

        # Backward pass on target class logit
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks failed to capture activations or gradients.")

        # Global average pool the gradients over spatial dimensions (H, W) -> shape (1, C, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Bilinear upsample to input dimensions (512, 512)
        cam = F.interpolate(cam, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            heatmap = (cam - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam)

        return heatmap, target_logit, target_prob

    def analyze_localization(
        self,
        img_tensor: torch.Tensor,
        grid: BreastGrid,
        class_idx: int = 2,
        label_family: str | None = None,
        activation_thresh: float = 0.50,
    ) -> LocalizationResult:
        """Compute Grad-CAM and attribute heat distribution over the anatomical grid."""
        heatmap, target_logit, target_prob = self.generate_heatmap(
            img_tensor, class_idx=class_idx, label_family=label_family
        )

        img_np = img_tensor[0, 0].cpu().numpy()
        tissue_mask = segment_breast_tissue(img_np)

        # Mask heatmap to tissue region
        masked_heat = heatmap * tissue_mask
        total_tissue_heat = float(np.sum(masked_heat))

        # Peak activation coordinate (y*, x*)
        if total_tissue_heat > 0:
            flat_idx = int(np.argmax(masked_heat))
            y_peak, x_peak = np.unravel_index(flat_idx, masked_heat.shape)
        else:
            flat_idx = int(np.argmax(heatmap))
            y_peak, x_peak = np.unravel_index(flat_idx, heatmap.shape)

        peak_coord = (int(y_peak), int(x_peak))

        # Find which cell contains the peak
        peak_cell_name = "central-middle"
        for cell in grid.cells:
            ymin, xmin, ymax, xmax = cell.bbox
            if ymin <= y_peak <= ymax and xmin <= x_peak <= xmax:
                peak_cell_name = cell.cell_name
                break

        # Calculate heat mass fraction per cell
        cell_heat_fractions = {}
        for cell in grid.cells:
            ymin, xmin, ymax, xmax = cell.bbox
            cell_heat = float(np.sum(masked_heat[ymin:ymax, xmin:xmax]))
            frac = cell_heat / max(1e-6, total_tissue_heat) if total_tissue_heat > 0 else 0.0
            cell_heat_fractions[cell.cell_name] = frac

        # Dominant cell with largest integrated heat mass
        dominant_cell_name = max(cell_heat_fractions, key=cell_heat_fractions.get)
        dominant_cell_fraction = float(cell_heat_fractions[dominant_cell_name])

        # Highlighted region area: pixels with activation >= activation_thresh * peak
        high_intensity_mask = (masked_heat >= activation_thresh) & tissue_mask
        highlighted_pixels = int(np.sum(high_intensity_mask))
        highlighted_pct = (highlighted_pixels / max(1, grid.total_tissue_pixels)) * 100.0

        return LocalizationResult(
            class_idx=class_idx,
            target_logit=target_logit,
            target_prob=target_prob,
            heatmap=heatmap,
            peak_coord=peak_coord,
            peak_cell_name=peak_cell_name,
            highlighted_area_pixels=highlighted_pixels,
            highlighted_area_pct_breast=highlighted_pct,
            cell_heat_fractions=cell_heat_fractions,
            dominant_cell_name=dominant_cell_name,
            dominant_cell_fraction=dominant_cell_fraction,
        )
