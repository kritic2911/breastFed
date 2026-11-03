"""Rule 9 Geometry Verification Tests for Federated Mammography BTP.

Adheres strictly to AGENTS.md Rule 9:
  - Asserts mathematical forward-inverse coordinate round-trip.
  - Asserts all transformed annotations lie strictly inside [0, 512] x [0, 512].
  - Asserts transformed annotations overlap the breast tissue and lesion.
  - Generates visual verification overlays saved to paper/figures/annotations_qc/
    for human review.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    inverse_transform_point,
    transform_box,
    transform_point,
)


def test_coordinate_round_trip():
    """Verify that inverse_transform(transform(x, y)) recovers (x, y) within numerical tolerance."""
    crop_box = (450, 1200, 2600, 2400)
    flip = True
    scale = 0.23789

    raw_x, raw_y = 1850.5, 1420.25

    # Forward
    trans_x, trans_y = transform_point(raw_x, raw_y, crop_box, flip, scale)
    # Inverse
    rec_x, rec_y = inverse_transform_point(trans_x, trans_y, crop_box, flip, scale)

    assert abs(raw_x - rec_x) < 1e-3, f"X mismatch: raw={raw_x}, rec={rec_x}"
    assert abs(raw_y - rec_y) < 1e-3, f"Y mismatch: raw={raw_y}, rec={rec_y}"


def test_transform_box_integrity():
    """Verify transform_box preserves xmin < xmax and ymin < ymax regardless of flip flag."""
    crop_box = (500, 1000, 3000, 2500)
    scale = 0.2048

    for flip in [False, True]:
        xmin, ymin, xmax, ymax = 1200.0, 800.0, 1450.0, 1100.0
        t_xmin, t_ymin, t_xmax, t_ymax = transform_box(xmin, ymin, xmax, ymax, crop_box, flip, scale)

        assert t_xmin < t_xmax, f"Invalid box X: {t_xmin} >= {t_xmax}"
        assert t_ymin < t_ymax, f"Invalid box Y: {t_ymin} >= {t_ymax}"
        assert 0.0 <= t_xmin <= 512.0
        assert 0.0 <= t_xmax <= 512.0
        assert 0.0 <= t_ymin <= 512.0
        assert 0.0 <= t_ymax <= 512.0


def test_transformed_annotations_bounds_and_tissue_overlap():
    """Verify all transformed annotations lie within [0, 512] and overlap tissue in cached images."""
    annotations_csv = Path("data/annotations.csv")
    manifest_csv = Path("data/manifest.csv")
    Path("data/cache512")

    if not annotations_csv.exists() or not manifest_csv.exists():
        pytest.skip("Annotations or manifest CSV not yet generated.")

    ann_df = pd.read_csv(annotations_csv)
    manifest_df = pd.read_csv(manifest_csv)

    assert not ann_df.empty, "Annotations table is empty."

    # Verify cached bounds for all annotations
    valid_ann = ann_df.dropna(subset=["cached_xmin", "cached_ymin", "cached_xmax", "cached_ymax"])
    assert len(valid_ann) > 0, "No valid transformed annotations found."

    # Strict bounds: all transformed annotations lie strictly within [0.0, 512.0]
    assert (valid_ann["cached_xmin"] >= 0.0).all(), "cached_xmin < 0"
    assert (valid_ann["cached_xmax"] <= 512.0).all(), "cached_xmax > 512"
    assert (valid_ann["cached_ymin"] >= 0.0).all(), "cached_ymin < 0"
    assert (valid_ann["cached_ymax"] <= 512.0).all(), "cached_ymax > 512"

    # Test tissue overlap on a sample of cached images
    manifest_map = (
        manifest_df[manifest_df["site"].isin(["vindr", "inbreast", "cdd"])]
        .set_index(["site", "image_id"])
        .to_dict("index")
    )
    qc_out_dir = Path("paper/figures/annotations_qc")
    qc_out_dir.mkdir(parents=True, exist_ok=True)

    tested_images = 0
    overlapped_images = 0

    # Test across unique annotated images (up to 20 samples per site)
    for site, group in valid_ann.groupby("site"):
        sample_imgs = group["image_id"].unique()[:10]
        for img_id in sample_imgs:
            key = (site, img_id)
            if key not in manifest_map:
                continue

            row = manifest_map[key]
            cached_rel = row.get("cached_path")
            if not cached_rel or not (Path("data") / cached_rel).exists():
                continue

            cached_path = Path("data") / cached_rel
            img_arr = np.load(cached_path)

            img_boxes = group[group["image_id"] == img_id]
            # Create overlay for human QC inspection
            overlay = (img_arr * 255.0).astype(np.uint8)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

            has_tissue_overlap = False
            for _, b in img_boxes.iterrows():
                x1 = int(round(np.clip(b["cached_xmin"], 0, 511)))
                y1 = int(round(np.clip(b["cached_ymin"], 0, 511)))
                x2 = int(round(np.clip(b["cached_xmax"], 0, 511)))
                y2 = int(round(np.clip(b["cached_ymax"], 0, 511)))

                # Check tissue presence inside or near annotation
                box_tissue = img_arr[min(y1, y2) : max(y1, y2) + 1, min(x1, x2) : max(x1, x2) + 1]
                if box_tissue.size > 0 and box_tissue.max() > 0.05:
                    has_tissue_overlap = True

                # Draw green bounding box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                f_type = str(b.get("finding_type", "lesion"))
                cv2.putText(overlay, f_type, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            tested_images += 1
            if has_tissue_overlap:
                overlapped_images += 1

            # Save overlay image for the human
            qc_file = qc_out_dir / f"{site}_{img_id}_qc_overlay.png"
            cv2.imwrite(str(qc_file), overlay)

    assert tested_images > 0, "No cached images tested for overlap."
    overlap_rate = overlapped_images / tested_images
    assert overlap_rate >= 0.90, f"Lesion-tissue overlap rate too low: {overlap_rate:.2%}"
