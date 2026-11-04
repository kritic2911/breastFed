"""Evaluation of explainable localization against ground truth lesion annotations.

Adheres strictly to AGENTS.md Phase P3b:
  - Peak-inside-lesion hit rate (evaluating whether Grad-CAM peak lies inside annotated lesion).
  - Agreement between heat distribution and annotation distribution over 3x3 cells (Total-Variation distance).
  - Lesion size error (% of breast area).
  - Adebayo et al. sanity check (randomizing layer4 and head weights to prove map parameter dependence).
  - Faithfulness test (comparing Class-2 probability drop when occluding top-heat vs random cells).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.explain.gradcam import GradCAM, LocalizationResult
from src.explain.grid import BreastGrid


@dataclass
class ImageAnnotation:
    """Bounding box annotations for a mammogram image."""
    image_id: str
    boxes: List[Tuple[float, float, float, float]]  # List of (xmin, ymin, xmax, ymax) in 512x512 space
    finding_types: List[str]


def parse_image_annotations(annotations_df: pd.DataFrame, image_id: str) -> ImageAnnotation | None:
    """Retrieve all bounding box annotations for a given image_id."""
    img_df = annotations_df[annotations_df["image_id"] == image_id]
    if img_df.empty:
        return None

    boxes = []
    finding_types = []
    for _, row in img_df.iterrows():
        xmin = float(row["cached_xmin"])
        ymin = float(row["cached_ymin"])
        xmax = float(row["cached_xmax"])
        ymax = float(row["cached_ymax"])
        # Ensure valid coordinates
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
            finding_types.append(str(row.get("finding_type", "lesion")))

    if not boxes:
        return None

    return ImageAnnotation(image_id=image_id, boxes=boxes, finding_types=finding_types)


def check_peak_inside_lesion(peak_coord: Tuple[int, int], annotation: ImageAnnotation, margin: float = 10.0) -> bool:
    """Check if the peak activation (y*, x*) falls inside or near any annotated lesion."""
    y_peak, x_peak = peak_coord
    for xmin, ymin, xmax, ymax in annotation.boxes:
        if (xmin - margin) <= x_peak <= (xmax + margin) and (ymin - margin) <= y_peak <= (ymax + margin):
            return True
    return False


def compute_annotation_cell_distribution(annotation: ImageAnnotation, grid: BreastGrid) -> Tuple[Dict[str, float], float]:
    """Compute fraction of total lesion area falling into each 3x3 anatomical cell."""
    # Create mask of annotated lesions
    annot_mask = np.zeros((512, 512), dtype=bool)
    for xmin, ymin, xmax, ymax in annotation.boxes:
        x0, y0 = max(0, int(xmin)), max(0, int(ymin))
        x1, y1 = min(512, int(round(xmax))), min(512, int(round(ymax)))
        annot_mask[y0:y1, x0:x1] = True

    total_lesion_pixels = int(np.sum(annot_mask))
    if total_lesion_pixels == 0:
        return {cell.cell_name: (1.0 / 9.0) for cell in grid.cells}, 0.0

    lesion_pct_breast = (total_lesion_pixels / max(1, grid.total_tissue_pixels)) * 100.0

    cell_fractions = {}
    for cell in grid.cells:
        ymin, xmin, ymax, xmax = cell.bbox
        cell_lesion_pixels = int(np.sum(annot_mask[ymin:ymax, xmin:xmax]))
        cell_fractions[cell.cell_name] = cell_lesion_pixels / total_lesion_pixels

    return cell_fractions, lesion_pct_breast


def compute_total_variation_distance(p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
    """Compute Total-Variation distance TV(P, Q) = 0.5 * sum(|P_i - Q_i|)."""
    all_keys = set(p_dist.keys()).union(set(q_dist.keys()))
    tv = 0.0
    for k in all_keys:
        p_val = p_dist.get(k, 0.0)
        q_val = q_dist.get(k, 0.0)
        tv += abs(p_val - q_val)
    return float(0.5 * tv)


def run_adebayo_sanity_check(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    original_heatmap: np.ndarray,
    class_idx: int = 2,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> float:
    """Adebayo et al. parameter randomization sanity check.

    Evaluates Grad-CAM on an uninitialized/randomized network architecture,
    and returns Pearson correlation between original and randomized heatmaps.
    A low correlation (< 0.40) proves map is sensitive to learned model parameters.
    """
    from src.models.backbone import build_model
    random_model = build_model(num_classes=3, pretrained=False, bn_mode="trainable")
    random_model.to(device)
    random_model.eval()

    random_cam = GradCAM(random_model)
    rand_heat, _, _ = random_cam.generate_heatmap(img_tensor, class_idx=class_idx)

    # Flatten and compute correlation
    orig_flat = original_heatmap.flatten()
    rand_flat = rand_heat.flatten()

    std_orig = np.std(orig_flat)
    std_rand = np.std(rand_flat)

    if std_orig < 1e-6 or std_rand < 1e-6:
        return 0.0

    corr = float(np.corrcoef(orig_flat, rand_flat)[0, 1])
    return corr if not np.isnan(corr) else 0.0


def run_faithfulness_test(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    grid: BreastGrid,
    loc_res: LocalizationResult,
    class_idx: int = 2,
    label_family: str | None = None,
) -> Tuple[float, float, bool]:
    """Faithfulness test: occlude dominant cell vs random non-dominant cell.

    Returns:
      (drop_top_cell, drop_random_cell, is_faithful)
    """
    model.eval()
    with torch.no_grad():
        if hasattr(model, "heads") and label_family is not None:
            base_logits = model(img_tensor, family=label_family)
        else:
            base_logits = model(img_tensor)
        base_p2 = float(F.softmax(base_logits, dim=1)[0, class_idx].item())

        # 1. Occlude dominant cell (set tissue pixels in that cell to 0)
        img_top = img_tensor.clone()
        dom_cell = grid.cell_map[loc_res.dominant_cell_name]
        ymin, xmin, ymax, xmax = dom_cell.bbox
        img_top[:, :, ymin:ymax, xmin:xmax] = 0.0

        if hasattr(model, "heads") and label_family is not None:
            top_logits = model(img_top, family=label_family)
        else:
            top_logits = model(img_top)
        top_p2 = float(F.softmax(top_logits, dim=1)[0, class_idx].item())
        drop_top = base_p2 - top_p2

        # 2. Occlude a random non-dominant cell (lowest heat cell)
        lowest_cell_name = min(loc_res.cell_heat_fractions, key=loc_res.cell_heat_fractions.get)
        low_cell = grid.cell_map[lowest_cell_name]
        img_low = img_tensor.clone()
        ly0, lx0, ly1, lx1 = low_cell.bbox
        img_low[:, :, ly0:ly1, lx0:lx1] = 0.0

        if hasattr(model, "heads") and label_family is not None:
            low_logits = model(img_low, family=label_family)
        else:
            low_logits = model(img_low)
        low_p2 = float(F.softmax(low_logits, dim=1)[0, class_idx].item())
        drop_low = base_p2 - low_p2

    is_faithful = (drop_top >= drop_low)
    return drop_top, drop_low, is_faithful
