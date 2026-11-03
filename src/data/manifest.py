"""Manifest generation and validation for federated mammography datasets.

Adheres strictly to AGENTS.md Section 1.1-1.3 & Rule 6:
  - Required columns: site, label_family, patient_hash, study, laterality, view,
    path, label, birads_raw, split, manufacturer, pixel_spacing,
    photometric_interpretation, age, density, crop_box, flip, scale.
  - Salted SHA-256 patient ID hashing (zero raw patient IDs in filenames/logs).
  - Caveats resolved per Section 1.2:
      * CMMD: exclude contralateral (~1,259 images); keep only lesion-breast images.
      * INbreast: exclude single 'FB' view.
      * CDD-CESM: low-energy DM only; exclude 4 unannotated & 34 indeterminable occult normal images.
      * VinDr: official patient-grouped 80/20 split; filter 0-byte stub files.
  - Stratified patient-grouped splits (test set ~20% of patients per site, 0 patient overlap).
"""

import glob
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split

from src.utils.provenance import hash_patient_id

logger = logging.getLogger(__name__)


def make_patient_grouped_split(
    df: pd.DataFrame,
    patient_col: str,
    label_col: str,
    test_size: float = 0.20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a stratified patient-grouped train/test split.

    Ensures 0 patient overlap and all views/breasts of each patient remain in the same split.
    """
    pt_labels = df.groupby(patient_col)[label_col].max()
    pt_ids = pt_labels.index.to_numpy()
    pt_y = pt_labels.to_numpy()

    # Stratified split by max patient diagnosis
    train_pts, test_pts = train_test_split(
        pt_ids, test_size=test_size, stratify=pt_y, random_state=random_state
    )

    test_set = set(test_pts)
    df = df.copy()
    df["split"] = df[patient_col].apply(lambda pid: "test" if pid in test_set else "train")
    return df


def parse_vindr(data_dir: Path) -> List[Dict[str, Any]]:
    """Parse VinDr-Mammo (mini)."""
    vindr_dir = data_dir / "vindr"
    if not vindr_dir.exists():
        logger.warning("VinDr directory not found: %s", vindr_dir)
        return []

    # Get valid non-empty DICOMs
    all_imgs = glob.glob(str(vindr_dir / "images" / "*" / "*.dicom"))
    valid_imgs = {Path(f).stem: f for f in all_imgs if Path(f).stat().st_size > 0}
    logger.info("VinDr: found %d valid non-empty DICOMs on disk", len(valid_imgs))

    breast_csv = vindr_dir / "breast-level_annotations.csv"
    if not breast_csv.exists():
        return []
    bdf = pd.read_csv(breast_csv)
    bdf = bdf[bdf["image_id"].isin(valid_imgs)].copy()

    # Metadata lookup
    meta_csv = vindr_dir / "metadata.csv"
    meta_dict = {}
    if meta_csv.exists():
        mdf = pd.read_csv(meta_csv)
        for _, row in mdf.iterrows():
            sop = str(row.get("SOP Instance UID", "")).strip()
            if sop in valid_imgs:
                meta_dict[sop] = row.to_dict()

    records = []
    for _, row in bdf.iterrows():
        img_id = str(row["image_id"]).strip()
        study_id = str(row["study_id"]).strip()
        path = valid_imgs[img_id]

        # 3-class label mapping:
        # 0 = Normal (BI-RADS 1)
        # 1 = Benign (BI-RADS 2, 3)
        # 2 = Suspicious/Malignant (BI-RADS 4, 5)
        raw_b = str(row["breast_birads"]).strip()
        if raw_b == "BI-RADS 1":
            label = 0
        elif raw_b in ["BI-RADS 2", "BI-RADS 3"]:
            label = 1
        elif raw_b in ["BI-RADS 4", "BI-RADS 5"]:
            label = 2
        else:
            continue

        # Split: official split column
        official_split = str(row.get("split", "training")).strip().lower()
        split = "train" if official_split == "training" else "test"

        # Metadata
        m_row = meta_dict.get(img_id, {})
        age_raw = str(m_row.get("Patient's Age", "")).strip()
        age = None
        if age_raw and any(c.isdigit() for c in age_raw):
            try:
                age = float("".join(c for c in age_raw if c.isdigit()))
            except ValueError:
                age = None

        mfg = str(m_row.get("Manufacturer's Model Name", m_row.get("Manufacturer", "Siemens"))).strip()
        photo = str(m_row.get("Photometric Interpretation", "MONOCHROME2")).strip()
        spacing = str(m_row.get("Imager Pixel Spacing", m_row.get("Pixel Spacing", ""))).strip()
        density = str(row.get("breast_density", "")).strip() or None

        records.append({
            "site": "vindr",
            "label_family": "birads",
            "patient_hash": hash_patient_id(study_id),
            "study": study_id,
            "image_id": img_id,
            "laterality": str(row["laterality"]).strip().upper(),
            "view": str(row["view_position"]).strip().upper(),
            "path": path,
            "label": label,
            "birads_raw": raw_b,
            "split": split,
            "source_format": "dicom",
            "bit_depth": 16,
            "manufacturer": mfg if mfg else "Siemens",
            "pixel_spacing": spacing if spacing else None,
            "photometric_interpretation": photo if photo else "MONOCHROME2",
            "age": age,
            "density": density,
        })

    return records


def parse_cmmd(data_dir: Path) -> List[Dict[str, Any]]:
    """Parse CMMD dataset."""
    cmmd_dir = data_dir / "cmmd" / "CMMD"
    if not cmmd_dir.exists():
        logger.warning("CMMD directory not found: %s", cmmd_dir)
        return []

    xlsx_files = list(cmmd_dir.glob("*.xlsx"))
    if not xlsx_files:
        return []
    df_clin = pd.read_excel(xlsx_files[0])

    # Build clinical map: patient_id -> dict of laterality -> details
    clin_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for _, row in df_clin.iterrows():
        pid = str(row["ID1"]).strip()
        lr = str(row["LeftRight"]).strip().upper()
        if pid not in clin_map:
            clin_map[pid] = {}
        clin_map[pid][lr] = {
            "abnormality": str(row.get("abnormality", "")).strip(),
            "classification": str(row.get("classification", "")).strip(),
            "age": row.get("Age"),
        }

    dcm_files = glob.glob(str(cmmd_dir / "**" / "*.dcm"), recursive=True)
    records = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue

        pid = str(getattr(ds, "PatientID", "")).strip()
        lat = str(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", ""))).strip().upper()

        # Caveat resolution: exclude contralateral (unlabeled) images
        if pid not in clin_map or lat not in clin_map[pid]:
            continue

        clin_info = clin_map[pid][lat]
        c_class = clin_info["classification"].lower()
        if "benign" in c_class:
            label = 1
        elif "malignant" in c_class:
            label = 2
        else:
            continue

        # View extraction
        view = getattr(ds, "ViewPosition", None)
        if not view and "ViewCodeSequence" in ds and len(ds.ViewCodeSequence) > 0:
            cm = ds.ViewCodeSequence[0].CodeMeaning.lower()
            if "oblique" in cm or "mlo" in cm:
                view = "MLO"
            elif "cranio" in cm or "cc" in cm:
                view = "CC"
        if not view:
            continue
        view = str(view).strip().upper()

        age = clin_info.get("age")
        try:
            age = float(age) if age is not None and not pd.isna(age) else None
        except (ValueError, TypeError):
            age = None

        mfg = str(getattr(ds, "Manufacturer", "GE Medical Systems")).strip() or "GE Medical Systems"
        spacing = str(getattr(ds, "ImagerPixelSpacing", getattr(ds, "PixelSpacing", "[0.09409, 0.09409]"))).strip()
        photo = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip()

        records.append({
            "site": "cmmd",
            "label_family": "biopsy",
            "patient_hash": hash_patient_id(pid),
            "study": pid,
            "image_id": Path(f).stem,
            "laterality": lat,
            "view": view,
            "path": f,
            "label": label,
            "birads_raw": None,
            "source_format": "dicom",
            "bit_depth": 8,
            "manufacturer": mfg,
            "pixel_spacing": spacing,
            "photometric_interpretation": photo,
            "age": age,
            "density": None,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = make_patient_grouped_split(df, patient_col="study", label_col="label", test_size=0.20, random_state=42)
        return df.to_dict("records")
    return []


def parse_inbreast(data_dir: Path) -> List[Dict[str, Any]]:
    """Parse INbreast dataset."""
    inb_dir = data_dir / "inbreast"
    if not inb_dir.exists():
        logger.warning("INbreast directory not found: %s", inb_dir)
        return []

    csv_path = inb_dir / "INbreast.csv"
    if not csv_path.exists():
        return []
    df_csv = pd.read_csv(csv_path, sep=";")

    dcm_files = glob.glob(str(inb_dir / "AllDICOMs" / "*.dcm"))
    file_map = {}
    for f in dcm_files:
        bn = Path(f).name
        parts = bn.split("_")
        if len(parts) >= 2:
            img_prefix = parts[0]
            pid = parts[1]
            file_map[img_prefix] = (f, pid)

    records = []
    for _, row in df_csv.iterrows():
        fname = str(row["File Name"]).strip()
        if fname not in file_map:
            continue

        raw_view = str(row["View"]).strip().upper()
        # Caveat resolution: exclude FB view per AGENTS.md Section 1.2
        if raw_view == "FB" or raw_view not in ["CC", "MLO"]:
            continue

        path, pid = file_map[fname]

        # 3-class label mapping:
        # 0 = Normal (BI-RADS 1)
        # 1 = Benign (BI-RADS 2, 3)
        # 2 = Suspicious/Malignant (BI-RADS 4a/b/c, 5, 6)
        raw_b = str(row["Bi-Rads"]).strip()
        if raw_b == "1":
            label = 0
        elif raw_b in ["2", "3"]:
            label = 1
        elif raw_b in ["4", "4a", "4b", "4c", "5", "6"]:
            label = 2
        else:
            continue

        age_val = row.get("Patient age")
        try:
            age = float(age_val) if age_val is not None and str(age_val).isdigit() else None
        except (ValueError, TypeError):
            age = None

        density_val = row.get("ACR")
        density = str(density_val).strip() if density_val is not None and not pd.isna(density_val) else None

        records.append({
            "site": "inbreast",
            "label_family": "birads",
            "patient_hash": hash_patient_id(pid),
            "study": pid,
            "image_id": fname,
            "laterality": str(row["Laterality"]).strip().upper(),
            "view": raw_view,
            "path": path,
            "label": label,
            "birads_raw": raw_b,
            "source_format": "dicom",
            "bit_depth": 16,
            "manufacturer": "Siemens",
            "pixel_spacing": "[0.07, 0.07]",
            "photometric_interpretation": "MONOCHROME2",
            "age": age,
            "density": density,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = make_patient_grouped_split(df, patient_col="study", label_col="label", test_size=0.20, random_state=42)
        return df.to_dict("records")
    return []


def parse_cdd(data_dir: Path) -> List[Dict[str, Any]]:
    """Parse CDD-CESM (low-energy DM images only)."""
    cdd_dir = data_dir / "cdd"
    if not cdd_dir.exists():
        logger.warning("CDD directory not found: %s", cdd_dir)
        return []

    excel_path = cdd_dir / "radiology_manual_ann.xlsx"
    if not excel_path.exists():
        return []
    df_all = pd.read_excel(excel_path, sheet_name="all")

    # Locate low energy images
    low_imgs = glob.glob(str(cdd_dir / "CDD-CESM" / "low_energy_images" / "*.jpg"))
    if not low_imgs:
        low_imgs = glob.glob(str(cdd_dir / "CDD-CESM" / "Low energy images of CDD-CESM" / "*.jpg"))

    disk_map = {Path(f).stem: f for f in low_imgs}
    df_dm = df_all[df_all["Image_name"].isin(disk_map)].copy()

    # Identify discordant occult normal images to exclude
    normal_tags = df_dm[df_dm["Tags"].astype(str).str.contains("normal", case=False)]
    discordant = normal_tags[normal_tags["Pathology Classification/ Follow up"].isin(["Benign", "Malignant"])]
    exclude_images = set()

    for _, row in discordant.iterrows():
        pid = row["Patient_ID"]
        side = row["Side"]
        companion = df_dm[(df_dm["Patient_ID"] == pid) & (df_dm["Side"] == side) & (df_dm["Image_name"] != row["Image_name"])]
        if companion.empty:
            exclude_images.add(row["Image_name"])
        else:
            c_tags = " ".join(companion["Tags"].astype(str)).lower()
            if "normal" in c_tags and not any(k in c_tags for k in ["mass", "calc", "asym", "distort", "imln"]):
                exclude_images.add(row["Image_name"])

    logger.info("CDD: excluding %d mammographically normal images with occult pathology", len(exclude_images))

    records = []
    for _, row in df_dm.iterrows():
        img_name = str(row["Image_name"]).strip()
        if img_name in exclude_images:
            continue

        raw_path = str(row["Pathology Classification/ Follow up"]).strip().lower()
        if raw_path == "normal":
            label = 0
        elif raw_path == "benign":
            label = 1
        elif raw_path == "malignant":
            label = 2
        else:
            continue

        pid = f"CDD_{row['Patient_ID']}"
        raw_view = str(row["View"]).strip().upper()
        if raw_view not in ["CC", "MLO"]:
            continue

        age_val = row.get("Age")
        try:
            age = float(age_val) if age_val is not None and not pd.isna(age_val) else None
        except (ValueError, TypeError):
            age = None

        density_val = row.get("Breast density (ACR)")
        density = str(density_val).strip() if density_val is not None and not pd.isna(density_val) else None

        records.append({
            "site": "cdd",
            "label_family": "biopsy",
            "patient_hash": hash_patient_id(pid),
            "study": pid,
            "image_id": img_name,
            "laterality": str(row["Side"]).strip().upper(),
            "view": raw_view,
            "path": disk_map[img_name],
            "label": label,
            "birads_raw": str(row.get("BIRADS", "")).strip() or None,
            "source_format": "jpg",
            "bit_depth": 8,
            "manufacturer": f"Machine_{row.get('Machine', 1)}",
            "pixel_spacing": None,
            "photometric_interpretation": "MONOCHROME2",
            "age": age,
            "density": density,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = make_patient_grouped_split(df, patient_col="study", label_col="label", test_size=0.20, random_state=42)
        return df.to_dict("records")
    return []


def build_manifest(data_dir: str | Path = "data", output_csv: str | Path | None = None) -> pd.DataFrame:
    """Build unified manifest across all 4 sites."""
    data_path = Path(data_dir)
    logger.info("Parsing datasets from %s ...", data_path)

    all_records: List[Dict[str, Any]] = []
    all_records.extend(parse_vindr(data_path))
    all_records.extend(parse_cmmd(data_path))
    all_records.extend(parse_inbreast(data_path))
    all_records.extend(parse_cdd(data_path))

    manifest_df = pd.DataFrame(all_records)
    logger.info("Parsed %d total records across %d sites", len(manifest_df), manifest_df["site"].nunique())

    # Initialize transform columns (filled during preprocessing)
    for col in ["crop_box", "flip", "scale", "cached_path", "original_shape", "tissue_pct"]:
        if col not in manifest_df.columns:
            manifest_df[col] = None

    if output_csv is not None:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(out_p, index=False)
        logger.info("Saved initial manifest to %s", out_p)

    return manifest_df

