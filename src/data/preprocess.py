"""Preprocessing and caching pipeline for digital mammography.

Adheres strictly to AGENTS.md Section 2.1 & Rule 9:
  - DICOM read (pydicom), VOI LUT / windowing, MONOCHROME1 vs MONOCHROME2 inversion.
  - Tissue segmentation mask, breast bounding box (crop_box), background/marker exclusion.
  - Standard laterality orientation: chest wall on left (posterior), nipple pointing right (anterior).
  - Per-image percentile normalization over tissue pixels (no CLAHE by default).
  - Aspect-preserving resize and padding to 512x512.
  - Exact forward and backward coordinate geometry transforms for lesion annotations.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import pydicom

try:
    from pydicom.pixels import apply_voi_lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_image_array(
    path: str | Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read raw mammogram from DICOM or standard image format (JPEG, PNG).

    Adheres strictly to AGENTS.md Section 2.1:
      - DICOM: pydicom read; apply VOI LUT / windowing; handle MONOCHROME1 vs MONOCHROME2;
        extract bit depth (no naive uint8 cast).
      - JPG/PNG: read 16-bit PNGs at full depth (cv2.IMREAD_UNCHANGED); convert color to grayscale;
        detect inversion from corner pixels (bright background -> inverted); extract bit depth.
      - Per-site list of file types: unreadable or unexpected formats fail loudly.
      - Converts every source to float [0.0, 1.0] before shared preprocessing.

    Returns:
        arr: 2D float32 image array in [0.0, 1.0] with MONOCHROME2 polarity (bright tissue, dark background).
        info: Dictionary containing metadata:
            - source_format: 'dicom', 'jpg', 'png', etc.
            - bit_depth: int (8, 12, 14, 16)
            - photometric: 'MONOCHROME2' or 'MONOCHROME1'
            - manufacturer: str
            - laterality: str ('L', 'R', or 'Unknown')
            - pixel_spacing: Optional[Tuple[float, float]]
            - lossy: bool
            - is_inverted: bool
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix in [".dcm", ".dicom"]:
        ds = pydicom.dcmread(str(path_obj), force=True)
        raw = ds.pixel_array.astype(np.float32)
        photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip().upper()
        manufacturer = str(getattr(ds, "Manufacturer", "Unknown")).strip()
        laterality = str(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", "Unknown"))).strip().upper()
        bit_depth = int(getattr(ds, "BitsStored", getattr(ds, "BitsAllocated", 16)))

        # Extract pixel spacing
        spacing = getattr(ds, "PixelSpacing", getattr(ds, "ImagerPixelSpacing", None))
        pixel_spacing = None
        if spacing is not None and len(spacing) >= 2:
            try:
                pixel_spacing = (float(spacing[0]), float(spacing[1]))
            except (ValueError, TypeError):
                pixel_spacing = None

        # Apply VOI LUT (Window Center/Width) if present
        try:
            arr_lut = apply_voi_lut(ds.pixel_array, ds)
            arr = arr_lut.astype(np.float32)
        except Exception:
            arr = raw

        # Invert MONOCHROME1 so tissue is bright and air/background is dark
        is_inverted = False
        if photometric == "MONOCHROME1":
            arr = arr.max() - arr
            is_inverted = True

        # Normalize to float [0.0, 1.0]
        val_min = float(arr.min())
        val_max = float(arr.max())
        if val_max > val_min:
            arr = (arr - val_min) / (val_max - val_min)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        info = {
            "source_format": "dicom",
            "bit_depth": bit_depth,
            "photometric": photometric,
            "manufacturer": manufacturer,
            "laterality": laterality,
            "pixel_spacing": pixel_spacing,
            "lossy": False,
            "is_inverted": is_inverted,
        }
        return arr, info

    elif suffix in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
        # Standard image: cv2.IMREAD_UNCHANGED ensures 16-bit PNGs are read at full depth
        img = cv2.imread(str(path_obj), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at {path_obj}")

        # If 3 channels (e.g. RGB/BGR JPEG in CDD-CESM), convert to single channel grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype == np.uint16:
            bit_depth = 16
            max_val = 65535.0
        elif img.dtype == np.uint8:
            bit_depth = 8
            max_val = 255.0
        else:
            bit_depth = 8
            max_val = float(img.max()) if img.max() > 0 else 1.0

        # Detect inversion from pixel corners (bright background means inverted MONOCHROME1)
        h, w = img.shape[:2]
        corner_sz = max(5, min(h, w) // 20)
        corners = np.concatenate([
            img[:corner_sz, :corner_sz].ravel(),
            img[:corner_sz, -corner_sz:].ravel(),
            img[-corner_sz:, :corner_sz].ravel(),
            img[-corner_sz:, -corner_sz:].ravel(),
        ])
        corner_median = float(np.median(corners))
        is_inverted = corner_median > (0.5 * max_val)

        arr = img.astype(np.float32)
        if is_inverted:
            arr = max_val - arr
            photometric = "MONOCHROME1"
        else:
            photometric = "MONOCHROME2"

        # Normalize to float [0.0, 1.0]
        val_min = float(arr.min())
        val_max = float(arr.max())
        if val_max > val_min:
            arr = (arr - val_min) / (val_max - val_min)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        source_fmt = suffix.lstrip(".").lower()
        if source_fmt == "jpeg":
            source_fmt = "jpg"
        lossy = source_fmt in ["jpg", "jpeg"]

        info = {
            "source_format": source_fmt,
            "bit_depth": bit_depth,
            "photometric": photometric,
            "manufacturer": "Unknown",
            "laterality": "Unknown",
            "pixel_spacing": None,
            "lossy": lossy,
            "is_inverted": is_inverted,
        }
        return arr, info

    else:
        raise ValueError(
            f"Unsupported file format '{suffix}' for {path_obj}. "
            "Expected DICOM (.dcm, .dicom) or standard image (.jpg, .jpeg, .png)."
        )


def segment_breast_mask(
    arr: np.ndarray,
) -> Tuple[Tuple[int, int, int, int], bool, np.ndarray]:
    """Segment breast tissue mask, compute bounding box, and detect chest wall orientation.

    Args:
        arr: 2D float32 image array.

    Returns:
        crop_box: (ymin, xmin, ymax, xmax) bounding box of breast tissue in raw coordinates.
        chest_on_right: True if chest wall is on the right border (needs horizontal flip).
        tissue_mask: 2D uint8 mask (1 = tissue, 0 = background/marker).
    """
    h, w = arr.shape
    # Downscale for robust and fast segmentation
    scale = 512.0 / max(h, w)
    small_w = max(16, int(round(w * scale)))
    small_h = max(16, int(round(h * scale)))
    small = cv2.resize(arr, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Separate breast tissue (both fatty and dense) from black air background
    val_min = float(small.min())
    val_max = float(small.max())
    rng = val_max - val_min

    # Threshold at 5% of dynamic range above background floor
    thresh_val = val_min + 0.05 * rng
    thresh = (small > thresh_val).astype(np.uint8) * 255

    # Morphological cleaning: remove small markers, fill small interior holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Extract largest connected component (the breast tissue)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    if num_labels > 1:
        # Label 0 is background; find largest component among labels >= 1
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask_small = (labels == largest_label).astype(np.uint8)
    else:
        mask_small = np.ones_like(thresh, dtype=np.uint8)

    # Upscale mask to original resolution
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Determine bounding box
    ys, xs = np.where(mask > 0)
    if len(ys) > 0:
        ymin, ymax = int(ys.min()), int(ys.max())
        xmin, xmax = int(xs.min()), int(xs.max())
    else:
        ymin, ymax, xmin, xmax = 0, h - 1, 0, w - 1

    # Detect chest wall side: chest wall is the vertical edge with higher tissue concentration
    left_tissue = float(mask[:, : max(1, w // 4)].sum())
    right_tissue = float(mask[:, -max(1, w // 4) :].sum())
    chest_on_right = right_tissue > left_tissue

    return (ymin, xmin, ymax, xmax), chest_on_right, mask


def transform_point(
    x: float,
    y: float,
    crop_box: Tuple[int, int, int, int],
    flip: bool,
    scale: float,
) -> Tuple[float, float]:
    """Forward coordinate transform: maps raw coordinate (x, y) to 512x512 space.

    Rule 9 Geometry requirement:
      1. Crop: x_c = x - xmin, y_c = y - ymin
      2. Flip: if flip, x_f = (W_crop - 1) - x_c
      3. Scale: x' = x_f * scale, y' = y_c * scale
    """
    ymin, xmin, ymax, xmax = crop_box
    w_c = xmax - xmin + 1

    x_c = x - xmin
    y_c = y - ymin

    if flip:
        x_c = (w_c - 1) - x_c

    x_trans = x_c * scale
    y_trans = y_c * scale
    return float(x_trans), float(y_trans)


def inverse_transform_point(
    x_trans: float,
    y_trans: float,
    crop_box: Tuple[int, int, int, int],
    flip: bool,
    scale: float,
) -> Tuple[float, float]:
    """Backward coordinate transform: maps 512x512 space back to raw coordinate (x, y)."""
    ymin, xmin, ymax, xmax = crop_box
    w_c = xmax - xmin + 1

    x_c = x_trans / scale
    y_c = y_trans / scale

    if flip:
        x_c = (w_c - 1) - x_c

    x = xmin + x_c
    y = ymin + y_c
    return float(x), float(y)


def transform_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    crop_box: Tuple[int, int, int, int],
    flip: bool,
    scale: float,
    canvas_size: int = 512,
) -> Tuple[float, float, float, float]:
    """Transform bounding box from raw image coordinates to 512x512 space.

    Clamps box coordinates to [0.0, canvas_size] to guarantee that all transformed
    annotations lie strictly within the canvas boundaries even when raw annotations
    extended slightly beyond the raw image frame or breast boundary (Rule 9).
    """
    x1, y1 = transform_point(xmin, ymin, crop_box, flip, scale)
    x2, y2 = transform_point(xmax, ymax, crop_box, flip, scale)
    t_xmin = min(x1, x2)
    t_ymin = min(y1, y2)
    t_xmax = max(x1, x2)
    t_ymax = max(y1, y2)

    # Clamp to [0.0, canvas_size]
    t_xmin = max(0.0, min(float(canvas_size), t_xmin))
    t_ymin = max(0.0, min(float(canvas_size), t_ymin))
    t_xmax = max(0.0, min(float(canvas_size), t_xmax))
    t_ymax = max(0.0, min(float(canvas_size), t_ymax))

    # Maintain non-zero box dimension if min == max
    if t_xmax <= t_xmin:
        t_xmax = min(float(canvas_size), t_xmin + 1.0)
    if t_ymax <= t_ymin:
        t_ymax = min(float(canvas_size), t_ymin + 1.0)

    return t_xmin, t_ymin, t_xmax, t_ymax


def preprocess_mammogram(
    arr: np.ndarray,
    target_size: int = 512,
    p_low_pct: float = 1.0,
    p_high_pct: float = 99.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Complete preprocessing pipeline for a single mammography image.

    Pipeline:
      1. Tissue segmentation & breast bounding box.
      2. Crop to breast bounding box (removing black margins and background markers).
      3. Standard orientation: flip horizontally if chest wall is on the right so that
         chest wall is always on the left (posterior) and nipple points right (anterior).
      4. Per-image percentile normalization over tissue pixels to [0.0, 1.0] (no CLAHE).
      5. Aspect-preserving resize and padding to target_size x target_size.

    Returns:
        cached_img: 2D float32 numpy array of shape (target_size, target_size).
        meta: Transform metadata dict (crop_box, flip, scale, original_shape, tissue_pct).
    """
    h_orig, w_orig = arr.shape
    crop_box, chest_on_right, tissue_mask = segment_breast_mask(arr)
    ymin, xmin, ymax, xmax = crop_box

    # Crop tissue and mask
    cropped = arr[ymin : ymax + 1, xmin : xmax + 1].copy()
    cropped_mask = tissue_mask[ymin : ymax + 1, xmin : xmax + 1].copy()

    # Standard orientation: chest wall on left (flip if chest on right)
    flip = bool(chest_on_right)
    if flip:
        cropped = np.fliplr(cropped)
        cropped_mask = np.fliplr(cropped_mask)

    # Per-image percentile normalization over tissue pixels
    tissue_vals = cropped[cropped_mask > 0]
    if len(tissue_vals) > 50:
        p_low = float(np.percentile(tissue_vals, p_low_pct))
        p_high = float(np.percentile(tissue_vals, p_high_pct))
    else:
        p_low = float(np.percentile(cropped, p_low_pct))
        p_high = float(np.percentile(cropped, p_high_pct))

    denom = p_high - p_low if (p_high - p_low) > 1e-4 else 1.0
    norm_cropped = np.clip((cropped - p_low) / denom, 0.0, 1.0)
    # Zero out non-tissue background pixels outside mask
    norm_cropped[cropped_mask == 0] = 0.0

    # Aspect-preserving resize
    h_crop, w_crop = norm_cropped.shape
    scale = float(target_size) / max(h_crop, w_crop)
    new_w = min(target_size, max(1, int(round(w_crop * scale))))
    new_h = min(target_size, max(1, int(round(h_crop * scale))))

    resized = cv2.resize(norm_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to target_size x target_size (chest wall flush with left border x=0)
    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    canvas[:new_h, :new_w] = resized

    meta = {
        "crop_box": crop_box,
        "crop_box_str": f"{ymin},{xmin},{ymax},{xmax}",
        "flip": flip,
        "scale": round(scale, 6),
        "original_shape": (h_orig, w_orig),
        "original_shape_str": f"{h_orig},{w_orig}",
        "tissue_pct": round(float(tissue_mask.mean() * 100.0), 2),
    }

    return canvas, meta
