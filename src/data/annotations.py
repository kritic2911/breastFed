"""Lesion annotations parsing, geometry mapping, and validation.

Adheres strictly to AGENTS.md Rule 9 & Section 1.1:
  - Parses lesion annotations across sites:
      * VinDr: finding bounding boxes (xmin, ymin, xmax, ymax).
      * INbreast: OsiriX plists for mass polygons, calcification points/clusters.
      * CDD-CESM: manual hand-drawn annotations (polygons, ellipses, circles).
      * CMMD: no lesion locations (confirmed empty).
  - Transforms raw annotations to 512x512 cached space using stored (crop_box, flip, scale).
  - Provides forward and backward coordinate geometry mapping.
"""

import json
import logging
import plistlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data.preprocess import transform_box
from src.utils.provenance import hash_patient_id

logger = logging.getLogger(__name__)


def parse_vindr_annotations(data_dir: Path, valid_image_ids: set) -> List[Dict[str, Any]]:
    """Parse VinDr lesion bounding boxes."""
    finding_csv = data_dir / "vindr" / "finding_annotations.csv"
    if not finding_csv.exists():
        return []

    df = pd.read_csv(finding_csv)
    # Keep rows for valid images with coordinates
    boxes_df = df[df["image_id"].isin(valid_image_ids) & df["xmin"].notna()].copy()

    records = []
    for _, row in boxes_df.iterrows():
        img_id = str(row["image_id"]).strip()
        study_id = str(row["study_id"]).strip()
        cat = str(row["finding_categories"]).strip()
        cat_lower = cat.lower()

        if "mass" in cat_lower and "calc" in cat_lower:
            f_type = "both"
        elif "mass" in cat_lower:
            f_type = "mass"
        elif "calc" in cat_lower:
            f_type = "calcification"
        elif "distortion" in cat_lower:
            f_type = "distortion"
        elif "asymmetry" in cat_lower:
            f_type = "asymmetry"
        else:
            f_type = "other"

        xmin = float(row["xmin"])
        ymin = float(row["ymin"])
        xmax = float(row["xmax"])
        ymax = float(row["ymax"])

        records.append({
            "site": "vindr",
            "image_id": img_id,
            "patient_hash": hash_patient_id(study_id),
            "study": study_id,
            "finding_type": f_type,
            "finding_name": cat,
            "raw_xmin": xmin,
            "raw_ymin": ymin,
            "raw_xmax": xmax,
            "raw_ymax": ymax,
            "raw_points_json": json.dumps([[xmin, ymin], [xmax, ymax]]),
        })

    return records


def parse_inbreast_annotations(data_dir: Path, valid_image_ids: set) -> List[Dict[str, Any]]:
    """Parse INbreast OsiriX plists."""
    xml_dir = data_dir / "inbreast" / "AllXML"
    if not xml_dir.exists():
        return []

    # Map image file prefix to patient ID
    dcm_files = list((data_dir / "inbreast" / "AllDICOMs").glob("*.dcm"))
    file_to_pid = {}
    for f in dcm_files:
        parts = f.name.split("_")
        if len(parts) >= 2:
            file_to_pid[parts[0]] = parts[1]

    records = []
    xml_files = list(xml_dir.glob("*.xml"))
    for f in xml_files:
        img_id = f.stem
        if img_id not in valid_image_ids:
            continue
        pid = file_to_pid.get(img_id, "Unknown")

        try:
            with open(f, "rb") as fp:
                pl = plistlib.load(fp)
        except Exception as e:
            logger.debug("Failed to read INbreast XML %s: %s", f, e)
            continue

        for img_entry in pl.get("Images", []):
            for roi in img_entry.get("ROIs", []):
                roi_name = str(roi.get("Name", "Finding")).strip()
                roi_lower = roi_name.lower()

                if "mass" in roi_lower:
                    f_type = "mass"
                elif "calc" in roi_lower:
                    f_type = "calcification"
                elif "distort" in roi_lower:
                    f_type = "distortion"
                elif "asym" in roi_lower or "spicul" in roi_lower:
                    f_type = "asymmetry"
                else:
                    f_type = "other"

                points_raw = roi.get("Point_px", [])
                if not points_raw:
                    continue

                pts = []
                for p_str in points_raw:
                    try:
                        clean = p_str.strip("() ")
                        coords = [float(x.strip()) for x in clean.split(",")]
                        if len(coords) >= 2:
                            pts.append([coords[0], coords[1]])
                    except Exception:
                        continue

                if not pts:
                    continue

                pts_arr = np.array(pts)
                xmin = float(pts_arr[:, 0].min())
                xmax = float(pts_arr[:, 0].max())
                ymin = float(pts_arr[:, 1].min())
                ymax = float(pts_arr[:, 1].max())

                records.append({
                    "site": "inbreast",
                    "image_id": img_id,
                    "patient_hash": hash_patient_id(pid),
                    "study": pid,
                    "finding_type": f_type,
                    "finding_name": roi_name,
                    "raw_xmin": xmin,
                    "raw_ymin": ymin,
                    "raw_xmax": xmax,
                    "raw_ymax": ymax,
                    "raw_points_json": json.dumps(pts),
                })

    return records


def parse_cdd_annotations(data_dir: Path, valid_image_ids: set) -> List[Dict[str, Any]]:
    """Parse CDD-CESM manual annotations from radiology_hand_drawn.csv."""
    ann_csv = data_dir / "cdd" / "radiology_hand_drawn.csv"
    if not ann_csv.exists():
        return []

    df = pd.read_csv(ann_csv)
    records = []

    for _, row in df.iterrows():
        fname = str(row["#filename"]).strip()
        stem = Path(fname).stem
        if stem not in valid_image_ids:
            continue

        shape_str = str(row.get("region_shape_attributes", "")).strip()
        if not shape_str or shape_str == "{}":
            continue

        try:
            shape_dict = json.loads(shape_str)
        except Exception:
            continue

        name = shape_dict.get("name", "")
        pts = []
        if name in ["polygon", "polyline"]:
            xs = shape_dict.get("all_points_x", [])
            ys = shape_dict.get("all_points_y", [])
            pts = [[float(x), float(y)] for x, y in zip(xs, ys)]
            if not pts:
                continue
            arr = np.array(pts)
            xmin, xmax = float(arr[:, 0].min()), float(arr[:, 0].max())
            ymin, ymax = float(arr[:, 1].min()), float(arr[:, 1].max())

        elif name == "circle":
            cx = float(shape_dict.get("cx", 0))
            cy = float(shape_dict.get("cy", 0))
            r = float(shape_dict.get("r", 0))
            xmin, xmax = cx - r, cx + r
            ymin, ymax = cy - r, cy + r
            pts = [[cx, cy]]

        elif name == "ellipse":
            cx = float(shape_dict.get("cx", 0))
            cy = float(shape_dict.get("cy", 0))
            rx = float(shape_dict.get("rx", 0))
            ry = float(shape_dict.get("ry", 0))
            xmin, xmax = cx - rx, cx + rx
            ymin, ymax = cy - ry, cy + ry
            pts = [[cx, cy]]

        elif name == "point":
            cx = float(shape_dict.get("cx", 0))
            cy = float(shape_dict.get("cy", 0))
            xmin, xmax = cx - 5.0, cx + 5.0
            ymin, ymax = cy - 5.0, cy + 5.0
            pts = [[cx, cy]]

        else:
            continue

        pid = f"CDD_{stem.split('_')[0].replace('P', '')}"
        records.append({
            "site": "cdd",
            "image_id": stem,
            "patient_hash": hash_patient_id(pid),
            "study": pid,
            "finding_type": "lesion",
            "finding_name": f"{name}_annotation",
            "raw_xmin": xmin,
            "raw_ymin": ymin,
            "raw_xmax": xmax,
            "raw_ymax": ymax,
            "raw_points_json": json.dumps(pts),
        })

    return records


def build_annotations_table(
    manifest_df: pd.DataFrame,
    data_dir: str | Path = "data",
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Build unified lesion annotations table mapped into 512x512 cached coordinate space."""
    data_path = Path(data_dir)
    valid_ids_by_site = {
        site: set(manifest_df[manifest_df["site"] == site]["image_id"])
        for site in manifest_df["site"].unique()
    }

    raw_annotations: List[Dict[str, Any]] = []
    if "vindr" in valid_ids_by_site:
        raw_annotations.extend(parse_vindr_annotations(data_path, valid_ids_by_site["vindr"]))
    if "inbreast" in valid_ids_by_site:
        raw_annotations.extend(parse_inbreast_annotations(data_path, valid_ids_by_site["inbreast"]))
    if "cdd" in valid_ids_by_site:
        raw_annotations.extend(parse_cdd_annotations(data_path, valid_ids_by_site["cdd"]))

    ann_df = pd.DataFrame(raw_annotations)
    logger.info("Parsed %d raw annotations across sites", len(ann_df))

    if ann_df.empty:
        return ann_df

    # Merge with manifest to retrieve transform parameters (crop_box, flip, scale)
    meta_cols = ["image_id", "site", "crop_box", "flip", "scale", "split"]
    merged = ann_df.merge(manifest_df[meta_cols], on=["image_id", "site"], how="left")

    # Apply transform_box to compute cached coordinates
    cached_boxes = []
    for _, row in merged.iterrows():
        crop_box = row.get("crop_box")
        flip = row.get("flip")
        scale = row.get("scale")

        if crop_box is None or pd.isna(flip) or scale is None or pd.isna(scale):
            cached_boxes.append((np.nan, np.nan, np.nan, np.nan))
            continue

        if isinstance(crop_box, str):
            c_vals = [int(v.strip()) for v in crop_box.split(",")]
            crop_tuple = (c_vals[0], c_vals[1], c_vals[2], c_vals[3])
        else:
            crop_tuple = crop_box

        c_xmin, c_ymin, c_xmax, c_ymax = transform_box(
            row["raw_xmin"],
            row["raw_ymin"],
            row["raw_xmax"],
            row["raw_ymax"],
            crop_tuple,
            bool(flip),
            float(scale),
        )
        cached_boxes.append((round(c_xmin, 2), round(c_ymin, 2), round(c_xmax, 2), round(c_ymax, 2)))

    boxes_arr = np.array(cached_boxes)
    ann_df["cached_xmin"] = boxes_arr[:, 0]
    ann_df["cached_ymin"] = boxes_arr[:, 1]
    ann_df["cached_xmax"] = boxes_arr[:, 2]
    ann_df["cached_ymax"] = boxes_arr[:, 3]
    ann_df["split"] = merged["split"]

    if output_csv is not None:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        ann_df.to_csv(out_p, index=False)
        logger.info("Saved annotations table to %s", out_p)

    return ann_df

