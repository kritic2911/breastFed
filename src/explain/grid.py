"""Classical anatomical spatial partitioning and 3x3 breast grid measurements.

Adheres strictly to AGENTS.md Phase P3b:
  - Breast area measured from segmentation mask over tissue pixels.
  - Standard orientation: chest wall on left (posterior), nipple on right (anterior).
  - 3x3 grid fitted to breast bounding box:
      * Rows: Upper / Central / Lower (MLO view) or Lateral / Central / Medial (CC view).
      * Columns: Posterior (chest wall) / Middle / Anterior (nipple).
  - Calculates cell coordinates, cell boundaries, and tissue area percentages.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import binary_fill_holes, label


@dataclass
class GridCell:
    """Represents one cell in the 3x3 anatomical grid."""
    row_idx: int
    col_idx: int
    row_name: str
    col_name: str
    cell_name: str  # e.g., "upper-posterior" or "lateral-middle"
    bbox: Tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax) in 512x512 space
    tissue_pixel_count: int
    fraction_of_breast: float  # Fraction of total breast tissue area inside this cell


@dataclass
class BreastGrid:
    """Anatomical 3x3 grid and spatial measurements for a single breast image."""
    view: str  # "CC" or "MLO"
    breast_bbox: Tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax)
    total_tissue_pixels: int
    breast_fraction_of_image: float  # Fraction of 512x512 image occupied by tissue
    cells: List[GridCell]
    cell_map: Dict[str, GridCell]


def segment_breast_tissue(img: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Segment breast tissue from background in normalized [0, 1] 512x512 image."""
    binary = img > threshold
    filled = binary_fill_holes(binary)

    # Retain the largest connected component (breast profile)
    labeled_array, num_features = label(filled)
    if num_features == 0:
        return binary

    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # ignore background
    largest_comp = int(np.argmax(component_sizes))
    return labeled_array == largest_comp


def fit_anatomical_grid(img: np.ndarray, view: str = "MLO") -> BreastGrid:
    """Fit standardized 3x3 anatomical grid to breast tissue bounding box."""
    tissue_mask = segment_breast_tissue(img)
    total_pixels = int(np.sum(tissue_mask))

    h, w = img.shape[:2]
    total_img_pixels = h * w

    if total_pixels == 0:
        # Fallback if image is empty
        bbox = (0, 0, h, w)
        total_pixels = total_img_pixels
        tissue_mask = np.ones((h, w), dtype=bool)
    else:
        y_indices, x_indices = np.where(tissue_mask)
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
        # Ensure minimum dimension of 30 pixels
        ymax = max(ymax, ymin + 30)
        xmax = max(xmax, xmin + 30)
        bbox = (ymin, xmin, ymax, xmax)

    ymin, xmin, ymax, xmax = bbox
    row_height = (ymax - ymin) / 3.0
    col_width = (xmax - xmin) / 3.0

    # Naming convention per AGENTS.md Section 279
    is_cc = (view.upper() == "CC")
    row_names = ["lateral", "central", "medial"] if is_cc else ["upper", "central", "lower"]
    col_names = ["posterior", "middle", "anterior"]

    cells = []
    cell_map = {}

    for r in range(3):
        r_y0 = int(round(ymin + r * row_height))
        r_y1 = int(round(ymin + (r + 1) * row_height)) if r < 2 else ymax

        for c in range(3):
            c_x0 = int(round(xmin + c * col_width))
            c_x1 = int(round(xmin + (c + 1) * col_width)) if c < 2 else xmax

            cell_mask = tissue_mask[r_y0:r_y1, c_x0:c_x1]
            cell_pixels = int(np.sum(cell_mask))
            frac_breast = cell_pixels / max(1, total_pixels)

            r_name = row_names[r]
            c_name = col_names[c]
            c_tag = f"{r_name}-{c_name}"

            cell_obj = GridCell(
                row_idx=r,
                col_idx=c,
                row_name=r_name,
                col_name=c_name,
                cell_name=c_tag,
                bbox=(r_y0, c_x0, r_y1, c_x1),
                tissue_pixel_count=cell_pixels,
                fraction_of_breast=frac_breast,
            )
            cells.append(cell_obj)
            cell_map[c_tag] = cell_obj

    return BreastGrid(
        view=view.upper(),
        breast_bbox=bbox,
        total_tissue_pixels=total_pixels,
        breast_fraction_of_image=float(total_pixels / total_img_pixels),
        cells=cells,
        cell_map=cell_map,
    )
