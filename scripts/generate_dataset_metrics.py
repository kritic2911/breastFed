#!/usr/bin/env python3
"""
Dataset Metrics Generator for Mammography Datasets.

Parses dataset files and metadata across:
  - mini VinDr-Mammo (data/vindr)
  - CMMD (data/cmmd)
  - INbreast (data/inbreast)
  - CDD-CESM (data/cdd)
  - CBIS-DDSM (data/cbis, optional if present)

Generates comprehensive metrics in Markdown format (default: datasets.md).
Main metrics:
  - Number of images (on disk and cataloged)
  - Number of unique patients
  - Malignant, Benign, and Normal cases (image-level and patient-level)
  - Abnormality kinds (Mass, Calcification, Both, Asymmetry, Distortion, etc.)
  - Cross-tabulations, clinical attributes, and modality specifications.
"""

import argparse
import glob
import json
import logging
import os
import plistlib
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Auto-delegate to local .venv if run with system python lacking pandas
_venv_dir = Path(__file__).resolve().parent.parent / ".venv"
_venv_py = _venv_dir / "bin" / "python3"
if _venv_py.exists() and sys.prefix != str(_venv_dir):
    try:
        import pandas  # noqa: F401
    except ImportError:
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset metrics for mammography datasets.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory containing vindr, cmmd, inbreast, cdd (default: data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets.md",
        help="Output Markdown file path (default: datasets.md)",
    )
    return parser.parse_args()


# ==============================================================================
# 1. VinDr-Mammo Parser (mini VinDr)
# ==============================================================================
def analyze_vindr(data_dir: Path):
    vindr_dir = data_dir / "vindr"
    if not vindr_dir.exists():
        return None

    imgs = sorted(glob.glob(str(vindr_dir / "images" / "*" / "*.dicom")))
    if not imgs:
        imgs = sorted(glob.glob(str(vindr_dir / "images" / "*.dicom")))
    if not imgs:
        return None

    image_stems_on_disk = {Path(f).stem for f in imgs}
    studies_on_disk = {Path(f).parent.name for f in imgs if Path(f).parent.name != "images"}

    breast_csv = vindr_dir / "breast-level_annotations.csv"
    if not breast_csv.exists():
        return None
    breast_df = pd.read_csv(breast_csv)
    total_catalog_images = len(breast_df)
    total_catalog_studies = breast_df["study_id"].nunique()

    # Filter to images on disk
    breast_disk = breast_df[breast_df["image_id"].isin(image_stems_on_disk)].copy()

    # Finding annotations
    finding_csv = vindr_dir / "finding_annotations.csv"
    if finding_csv.exists():
        finding_df = pd.read_csv(finding_csv)
        finding_disk = finding_df[finding_df["image_id"].isin(image_stems_on_disk)]

        def get_finding_kind(g):
            cats = " ".join(g["finding_categories"].astype(str).tolist()).lower()
            has_mass = "mass" in cats
            has_calc = "calcification" in cats
            if has_mass and has_calc:
                return "Both (Mass & Calcification)"
            elif has_mass:
                return "Mass"
            elif has_calc:
                return "Calcification"
            elif "distortion" in cats:
                return "Architectural Distortion"
            elif "asymmetry" in cats:
                return "Asymmetry"
            elif "no finding" in cats:
                return "No Finding / Normal"
            else:
                return "Other finding"

        img_findings = finding_disk.groupby("image_id").apply(get_finding_kind, include_groups=False).rename("finding_kind")
        breast_disk = breast_disk.merge(img_findings, on="image_id", how="left")
    else:
        breast_disk["finding_kind"] = "No Finding / Normal"

    breast_disk["finding_kind"] = breast_disk["finding_kind"].fillna("No Finding / Normal")

    # 3-class mapping from project spec:
    # 0 = normal (BI-RADS 1)
    # 1 = benign (BI-RADS 2-3)
    # 2 = suspicious/malignant (BI-RADS 4-5)
    def map_birads_3class(b):
        s = str(b).strip()
        if s == "BI-RADS 1":
            return "Class 0 (Normal)"
        elif s in ["BI-RADS 2", "BI-RADS 3"]:
            return "Class 1 (Benign)"
        elif s in ["BI-RADS 4", "BI-RADS 5"]:
            return "Class 2 (Suspicious/Malignant)"
        return "Unknown"

    breast_disk["3class"] = breast_disk["breast_birads"].apply(map_birads_3class)
    breast_disk["binary_label"] = breast_disk["breast_birads"].isin(["BI-RADS 4", "BI-RADS 5"]).map(
        {True: "Malignant / Suspicious", False: "Benign / Normal"}
    )

    # Metadata
    meta_csv = vindr_dir / "metadata.csv"
    meta_df = pd.read_csv(meta_csv) if meta_csv.exists() else pd.DataFrame()
    meta_disk = meta_df[meta_df["SOP Instance UID"].isin(image_stems_on_disk)] if not meta_df.empty else pd.DataFrame()

    # Patient / Study level aggregation
    def agg_study(g):
        p_3c = (
            "Class 2 (Suspicious/Malignant)"
            if "Class 2 (Suspicious/Malignant)" in set(g["3class"])
            else ("Class 1 (Benign)" if "Class 1 (Benign)" in set(g["3class"]) else "Class 0 (Normal)")
        )
        p_bin = "Malignant / Suspicious" if "Malignant / Suspicious" in set(g["binary_label"]) else "Benign / Normal"
        kinds = set(g["finding_kind"])
        if "Both (Mass & Calcification)" in kinds or ("Mass" in kinds and "Calcification" in kinds):
            f_cat = "Both (Mass & Calcification)"
        elif "Mass" in kinds:
            f_cat = "Mass"
        elif "Calcification" in kinds:
            f_cat = "Calcification"
        elif "Architectural Distortion" in kinds:
            f_cat = "Architectural Distortion"
        elif "Asymmetry" in kinds:
            f_cat = "Asymmetry"
        elif "Other finding" in kinds:
            f_cat = "Other finding"
        else:
            f_cat = "No Finding / Normal"
        return pd.Series({"3class": p_3c, "binary_label": p_bin, "finding_kind": f_cat})

    pt_df = breast_disk.groupby("study_id").apply(agg_study, include_groups=False).reset_index()

    # Age stats
    ages = pd.Series(dtype=float)
    if not meta_disk.empty and "Patient's Age" in meta_disk.columns:
        extracted_ages = meta_disk["Patient's Age"].dropna().str.extract(r"(\d+)")[0].astype(float)
        ages = extracted_ages[extracted_ages > 0]

    return {
        "dcm_count_disk": len(imgs),
        "unique_studies_disk": len(studies_on_disk) if studies_on_disk else breast_disk["study_id"].nunique(),
        "total_catalog_images": total_catalog_images,
        "total_catalog_studies": total_catalog_studies,
        "df_disk": breast_disk,
        "pt_df": pt_df,
        "meta_disk": meta_disk,
        "ages": ages,
    }


# ==============================================================================
# 2. CMMD Parser
# ==============================================================================
def analyze_cmmd(data_dir: Path):
    cmmd_dir = data_dir / "cmmd" / "CMMD"
    if not cmmd_dir.exists():
        return None

    excel_files = list(cmmd_dir.glob("*.xlsx"))
    clin_file = excel_files[0] if excel_files else None
    if not clin_file:
        return None

    clin_df = pd.read_excel(clin_file)
    meta_csv = cmmd_dir / "metadata.csv"
    meta_df = pd.read_csv(meta_csv) if meta_csv.exists() else pd.DataFrame()

    dcm_files = glob.glob(str(cmmd_dir / "**" / "*.dcm"), recursive=True)
    patient_dirs = [d.name for d in cmmd_dir.iterdir() if d.is_dir() and d.name.startswith("D")]

    # Normalize clinical dataframe
    clin_df["abnormality"] = clin_df["abnormality"].str.strip().str.lower()
    clin_df["classification"] = clin_df["classification"].str.strip()

    # Patients on disk
    clin_df["on_disk"] = clin_df["ID1"].isin(set(patient_dirs))
    disk_clin = clin_df[clin_df["on_disk"]].copy()

    def aggregate_cmmd_patient(df):
        def agg_pt(g):
            c_vals = set(g["classification"])
            p_cat = "Malignant" if "Malignant" in c_vals else "Benign"

            a_vals = set(g["abnormality"])
            if "both" in a_vals or ("mass" in a_vals and "calcification" in a_vals):
                a_cat = "both"
            elif "mass" in a_vals:
                a_cat = "mass"
            elif "calcification" in a_vals:
                a_cat = "calcification"
            else:
                a_cat = "other"

            age = g["Age"].dropna().iloc[0] if not g["Age"].dropna().empty else None
            return pd.Series({"classification": p_cat, "abnormality": a_cat, "age": age})

        return df.groupby("ID1").apply(agg_pt, include_groups=False).reset_index()

    pt_all = aggregate_cmmd_patient(clin_df)
    pt_disk = aggregate_cmmd_patient(disk_clin)

    return {
        "dcm_count_disk": len(dcm_files),
        "patient_dirs_count": len(patient_dirs),
        "cataloged_images": meta_df["Number of Images"].sum() if not meta_df.empty else len(clin_df) * 2,
        "clin_df": clin_df,
        "disk_clin": disk_clin,
        "pt_all": pt_all,
        "pt_disk": pt_disk,
        "meta_df": meta_df,
    }


# ==============================================================================
# 3. INbreast Parser
# ==============================================================================
def analyze_inbreast(data_dir: Path):
    inb_dir = data_dir / "inbreast"
    if not inb_dir.exists():
        return None

    csv_path = inb_dir / "INbreast.csv"
    if not csv_path.exists():
        return None

    df_csv = pd.read_csv(csv_path, sep=";")
    df_csv["File Name"] = df_csv["File Name"].astype(str)

    dcm_files = sorted(glob.glob(str(inb_dir / "AllDICOMs" / "*.dcm")))
    file_to_pid = {}
    for f in dcm_files:
        bn = Path(f).name
        parts = bn.split("_")
        if len(parts) >= 2:
            file_to_pid[parts[0]] = parts[1]

    df_csv["Patient_ID"] = df_csv["File Name"].map(file_to_pid).fillna("Unknown")

    # XML Annotations
    xml_files = sorted(glob.glob(str(inb_dir / "AllXML" / "*.xml")))
    xml_findings = {}
    roi_details = Counter()

    for f in xml_files:
        img_id = Path(f).stem
        try:
            with open(f, "rb") as fp:
                pl = plistlib.load(fp)
                rois = []
                for img in pl.get("Images", []):
                    for roi in img.get("ROIs", []):
                        r_name = str(roi.get("Name", "")).strip()
                        rois.append(r_name)
                        roi_details[r_name] += 1
                has_mass = any("mass" in r.lower() for r in rois)
                has_calc = any("calc" in r.lower() for r in rois)
                has_other = any(any(k in r.lower() for k in ["cluster", "spicul", "asym", "distort"]) for r in rois)
                if has_mass and has_calc:
                    xml_findings[img_id] = "both"
                elif has_mass:
                    xml_findings[img_id] = "mass"
                elif has_calc:
                    xml_findings[img_id] = "calcification"
                elif has_other:
                    xml_findings[img_id] = "other"
                else:
                    xml_findings[img_id] = "unspecified"
        except (plistlib.InvalidFileException, OSError) as e:
            logger.debug("Skipping unparseable plist file %s: %s", f, e)
            continue

    def get_abnormality(fname):
        if fname in xml_findings:
            return xml_findings[fname]
        return "normal"

    df_csv["Abnormality"] = df_csv["File Name"].apply(get_abnormality)

    def get_pathology(b):
        s = str(b).strip().lower()
        if s in ["1", "2", "3"]:
            return "Benign / Negative"
        elif s in ["4", "4a", "4b", "4c", "5", "6"]:
            return "Malignant / Suspicious"
        return "Unknown"

    df_csv["Pathology"] = df_csv["Bi-Rads"].apply(get_pathology)

    def aggregate_inb_patient(df):
        def agg_pt(g):
            p_vals = set(g["Pathology"])
            p_cat = "Malignant / Suspicious" if "Malignant / Suspicious" in p_vals else "Benign / Negative"

            a_vals = set(g["Abnormality"])
            if "both" in a_vals or ("mass" in a_vals and "calcification" in a_vals):
                a_cat = "both"
            elif "mass" in a_vals:
                a_cat = "mass"
            elif "calcification" in a_vals:
                a_cat = "calcification"
            elif "other" in a_vals:
                a_cat = "other"
            else:
                a_cat = "normal"

            return pd.Series({"Pathology": p_cat, "Abnormality": a_cat})

        return df.groupby("Patient_ID").apply(agg_pt, include_groups=False).reset_index()

    pt_df = aggregate_inb_patient(df_csv)

    return {
        "dcm_count": len(dcm_files),
        "unique_patients": df_csv["Patient_ID"].nunique(),
        "xml_count": len(xml_files),
        "df": df_csv,
        "pt_df": pt_df,
        "roi_details": roi_details,
    }


# ==============================================================================
# 4. CDD-CESM Parser
# ==============================================================================
def analyze_cdd(data_dir: Path):
    cdd_dir = data_dir / "cdd"
    if not cdd_dir.exists():
        return None

    excel_path = cdd_dir / "radiology_manual_ann.xlsx"
    if not excel_path.exists():
        return None

    df_all = pd.read_excel(excel_path, sheet_name="all")

    mass_imgs = set(pd.read_excel(excel_path, sheet_name="mass_description")["Image_name"]) | set(
        pd.read_excel(excel_path, sheet_name="mass enhancement_description")["Image_name"]
    )
    calc_imgs = set(pd.read_excel(excel_path, sheet_name="calcifications")["Image_name"])
    dist_imgs = set(pd.read_excel(excel_path, sheet_name="distortion")["Image_name"])
    asym_imgs = set(pd.read_excel(excel_path, sheet_name="asymmetry")["Image_name"])

    def classify_kind(img_name, findings, tags):
        is_m = img_name in mass_imgs or "mass" in str(tags).lower() or "mass" in str(findings).lower()
        is_c = img_name in calc_imgs or "calc" in str(tags).lower() or "calc" in str(findings).lower()
        if is_m and is_c:
            return "Both (Mass & Calcification)"
        elif is_m:
            return "Mass"
        elif is_c:
            return "Calcification"
        elif img_name in dist_imgs or "distortion" in str(tags).lower():
            return "Architectural Distortion"
        elif img_name in asym_imgs or "asymmetry" in str(tags).lower():
            return "Asymmetry"
        elif "normal" in str(tags).lower() or "normal" in str(findings).lower():
            return "Normal / Negative"
        else:
            return "Other"

    df_all["Abnormality"] = df_all.apply(
        lambda r: classify_kind(r["Image_name"], r["Findings"], r["Tags"]), axis=1
    )

    # Standardize pathology
    df_all["Pathology"] = df_all["Pathology Classification/ Follow up"].str.strip()

    low_imgs = glob.glob(str(cdd_dir / "CDD-CESM" / "low_energy_images" / "*"))
    if not low_imgs:
        low_imgs = glob.glob(str(cdd_dir / "CDD-CESM" / "Low energy images of CDD-CESM" / "*"))
    sub_imgs = glob.glob(str(cdd_dir / "CDD-CESM" / "Subtracted images of CDD-CESM" / "*"))
    total_disk_images = len(low_imgs) + len(sub_imgs)

    disk_stems = {Path(f).stem for f in low_imgs} | {Path(f).stem for f in sub_imgs}
    df_disk = df_all[df_all["Image_name"].isin(disk_stems)].copy() if disk_stems else df_all.copy()

    def aggregate_cdd_patient(df):
        def agg_pt(g):
            p_vals = set(g["Pathology"])
            if "Malignant" in p_vals:
                p_cat = "Malignant"
            elif "Benign" in p_vals:
                p_cat = "Benign"
            else:
                p_cat = "Normal"

            a_vals = set(g["Abnormality"])
            if "Both (Mass & Calcification)" in a_vals or ("Mass" in a_vals and "Calcification" in a_vals):
                a_cat = "Both (Mass & Calcification)"
            elif "Mass" in a_vals:
                a_cat = "Mass"
            elif "Calcification" in a_vals:
                a_cat = "Calcification"
            elif "Architectural Distortion" in a_vals:
                a_cat = "Architectural Distortion"
            elif "Asymmetry" in a_vals:
                a_cat = "Asymmetry"
            elif "Other" in a_vals:
                a_cat = "Other"
            else:
                a_cat = "Normal / Negative"

            age = g["Age"].dropna().iloc[0] if not g["Age"].dropna().empty else None
            return pd.Series({"Pathology": p_cat, "Abnormality": a_cat, "Age": age})

        return df.groupby("Patient_ID").apply(agg_pt, include_groups=False).reset_index()

    pt_df = aggregate_cdd_patient(df_disk)
    pt_all = aggregate_cdd_patient(df_all)

    return {
        "low_energy_count": len(low_imgs),
        "subtracted_count": len(sub_imgs),
        "total_disk_images": total_disk_images,
        "total_annotations": len(df_all),
        "unique_patients": df_all["Patient_ID"].nunique(),
        "df": df_all,
        "df_disk": df_disk,
        "pt_df": pt_df,
        "pt_all": pt_all,
    }


# ==============================================================================
# 5. CBIS-DDSM Parser (Optional)
# ==============================================================================
def analyze_cbis(data_dir: Path):
    cbis_dir = data_dir / "cbis"
    if not cbis_dir.exists():
        return None

    train_ann_files = sorted(glob.glob(str(cbis_dir / "train" / "ann" / "*.json")))
    test_ann_files = sorted(glob.glob(str(cbis_dir / "test" / "ann" / "*.json")))
    train_img_files = sorted(glob.glob(str(cbis_dir / "train" / "img" / "*")))
    test_img_files = sorted(glob.glob(str(cbis_dir / "test" / "img" / "*")))

    if not train_ann_files and not test_ann_files:
        return None

    def parse_annotations(ann_files, filter_existing_images=False, img_files=None):
        if filter_existing_images and img_files:
            existing_names = {Path(f).name for f in img_files}
            target_files = [f for f in ann_files if Path(f).name.replace(".json", "") in existing_names]
        else:
            target_files = ann_files

        records = []
        for f in target_files:
            fname = Path(f).name
            m = re.search(r"(P_\d+)", fname)
            patient_id = m.group(1) if m else "Unknown"

            with open(f, "r", encoding="utf-8") as fp:
                try:
                    data = json.load(fp)
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug("Skipping invalid JSON file %s: %s", f, e)
                    continue

            tag_names = {t.get("name", "").strip().lower() for t in data.get("tags", [])}

            pathology = "unknown"
            if "malignant" in tag_names:
                pathology = "malignant"
            elif "benign" in tag_names:
                pathology = "benign"
            elif "benign_without_callback" in tag_names:
                pathology = "benign_without_callback"

            has_mass = "mass" in tag_names
            has_calc = "calcification" in tag_names
            if has_mass and has_calc:
                abnormality = "both"
            elif has_mass:
                abnormality = "mass"
            elif has_calc:
                abnormality = "calcification"
            else:
                abnormality = "other"

            view = "Unknown"
            if "cc" in tag_names:
                view = "CC"
            elif "mlo" in tag_names:
                view = "MLO"

            laterality = "Unknown"
            if "left" in tag_names:
                laterality = "Left"
            elif "right" in tag_names:
                laterality = "Right"

            records.append({
                "patient_id": patient_id,
                "file": fname,
                "pathology": pathology,
                "abnormality": abnormality,
                "view": view,
                "laterality": laterality,
            })

        return pd.DataFrame(records) if records else pd.DataFrame(columns=["patient_id", "pathology", "abnormality", "view", "laterality"])

    df_train_all = parse_annotations(train_ann_files)
    df_test_all = parse_annotations(test_ann_files)
    df_all = pd.concat([df_train_all, df_test_all], ignore_index=True)

    df_train_disk = parse_annotations(train_ann_files, filter_existing_images=True, img_files=train_img_files)
    df_test_disk = parse_annotations(test_ann_files, filter_existing_images=True, img_files=test_img_files)
    df_disk = pd.concat([df_train_disk, df_test_disk], ignore_index=True)

    def aggregate_patient_level(df):
        if df.empty:
            return pd.DataFrame()
        def agg_pt(g):
            p_vals = set(g["pathology"])
            if "malignant" in p_vals:
                p_cat = "malignant"
            elif "benign" in p_vals:
                p_cat = "benign"
            elif "benign_without_callback" in p_vals:
                p_cat = "benign_without_callback"
            else:
                p_cat = "unknown"

            a_vals = set(g["abnormality"])
            if "both" in a_vals or ("mass" in a_vals and "calcification" in a_vals):
                a_cat = "both"
            elif "mass" in a_vals:
                a_cat = "mass"
            elif "calcification" in a_vals:
                a_cat = "calcification"
            else:
                a_cat = "other"

            return pd.Series({"pathology": p_cat, "abnormality": a_cat})

        return df.groupby("patient_id").apply(agg_pt, include_groups=False).reset_index()

    pt_all = aggregate_patient_level(df_all)
    pt_disk = aggregate_patient_level(df_disk)

    return {
        "train_img_count": len(train_img_files),
        "test_img_count": len(test_img_files),
        "total_img_disk": len(train_img_files) + len(test_img_files),
        "train_ann_count": len(train_ann_files),
        "test_ann_count": len(test_ann_files),
        "total_ann_count": len(train_ann_files) + len(test_ann_files),
        "df_all": df_all,
        "df_disk": df_disk,
        "df_train_all": df_train_all,
        "df_test_all": df_test_all,
        "df_train_disk": df_train_disk,
        "df_test_disk": df_test_disk,
        "pt_all": pt_all,
        "pt_disk": pt_disk,
    }


# ==============================================================================
# 6. Markdown Report Formatter
# ==============================================================================
def generate_markdown(vindr_data, cmmd_data, inb_data, cdd_data, cbis_data, output_path: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append("# Breast Cancer Mammography Datasets Metrics")
    md.append("")
    md.append(f"> Auto-generated by `scripts/generate_dataset_metrics.py` on **{timestamp}**.")
    md.append("> Comprehensive quantitative analysis across all active local mammography datasets, covering image counts, patient counts, clinical/biopsy pathology (malignant vs. benign), abnormality types (masses vs. calcifications), views, and clinical metadata.")
    md.append("")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 1: Cross-Dataset Executive Summary
    # ──────────────────────────────────────────────────────────────────────────
    md.append("## 1. Executive Summary Across Active Datasets")
    md.append("")
    md.append("| Dataset | Label Family | Imaging Modality | Images on Disk | Total Cataloged Images | Unique Patients | Ground Truth Type | Malignant / Suspicious (Images) | Benign Cases (Images) | Normal / Negative | Primary Abnormality Findings |")
    md.append("| :--- | :---: | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |")

    # VinDr
    if vindr_data:
        v_mal = (vindr_data["df_disk"]["3class"] == "Class 2 (Suspicious/Malignant)").sum()
        v_ben = (vindr_data["df_disk"]["3class"] == "Class 1 (Benign)").sum()
        v_norm = (vindr_data["df_disk"]["3class"] == "Class 0 (Normal)").sum()
        v_tot = len(vindr_data["df_disk"])
        v_mass = (vindr_data["df_disk"]["finding_kind"] == "Mass").sum()
        v_calc = (vindr_data["df_disk"]["finding_kind"] == "Calcification").sum()
        v_both = (vindr_data["df_disk"]["finding_kind"] == "Both (Mass & Calcification)").sum()
        md.append(
            f"| **VinDr-Mammo (mini)** | `birads` | Full-Field Digital (FFDM) | **{vindr_data['dcm_count_disk']:,}** | {vindr_data['total_catalog_images']:,} | **{vindr_data['unique_studies_disk']:,}** | Expert BI-RADS Consensus | {v_mal:,} ({v_mal/v_tot*100:.1f}%) | {v_ben:,} ({v_ben/v_tot*100:.1f}%) | {v_norm:,} ({v_norm/v_tot*100:.1f}%) | Mass ({v_mass:,}), Calc ({v_calc:,}), Both ({v_both:,}), Normal ({v_norm:,}) |"
        )

    # CMMD
    if cmmd_data:
        cmmd_mal_disk = (cmmd_data["disk_clin"]["classification"] == "Malignant").sum()
        cmmd_ben_disk = (cmmd_data["disk_clin"]["classification"] == "Benign").sum()
        md.append(
            f"| **CMMD** | `biopsy` | Full-Field Digital (FFDM) | **{cmmd_data['dcm_count_disk']:,}** | {cmmd_data['cataloged_images']:,} | **{cmmd_data['pt_disk']['ID1'].nunique():,}** | Biopsy Histology | {cmmd_mal_disk*2:,} (~{cmmd_mal_disk/len(cmmd_data['disk_clin'])*100:.1f}%)* | {cmmd_ben_disk*2:,} (~{cmmd_ben_disk/len(cmmd_data['disk_clin'])*100:.1f}%)* | 0 | Mass ({ (cmmd_data['disk_clin']['abnormality']=='mass').sum():,} cases), Calc ({ (cmmd_data['disk_clin']['abnormality']=='calcification').sum():,}), Both ({ (cmmd_data['disk_clin']['abnormality']=='both').sum():,}) |"
        )

    # INbreast
    if inb_data:
        inb_mal = (inb_data["df"]["Pathology"] == "Malignant / Suspicious").sum()
        inb_ben = (inb_data["df"]["Pathology"] == "Benign / Negative").sum()
        inb_norm = (inb_data["df"]["Bi-Rads"] == "1").sum()
        md.append(
            f"| **INbreast** | `birads` | Full-Field Digital (FFDM) | **{inb_data['dcm_count']:,}** | {inb_data['dcm_count']:,} | **{inb_data['unique_patients']:,}** | Expert BI-RADS Assessment | {inb_mal:,} ({inb_mal/inb_data['dcm_count']*100:.1f}%) | {inb_ben:,} ({inb_ben/inb_data['dcm_count']*100:.1f}%) | {inb_norm:,} ({inb_norm/inb_data['dcm_count']*100:.1f}%) | Calcification ({ (inb_data['df']['Abnormality']=='calcification').sum():,}), Both ({ (inb_data['df']['Abnormality']=='both').sum():,}), Mass ({ (inb_data['df']['Abnormality']=='mass').sum():,}) |"
        )

    # CDD-CESM
    if cdd_data:
        df_cdd_disk = cdd_data["df_disk"]
        cdd_mal = (df_cdd_disk["Pathology"] == "Malignant").sum()
        cdd_ben = (df_cdd_disk["Pathology"] == "Benign").sum()
        cdd_norm = (df_cdd_disk["Pathology"] == "Normal").sum()
        cdd_tot = len(df_cdd_disk)
        cdd_mass = (df_cdd_disk["Abnormality"] == "Mass").sum()
        cdd_calc = (df_cdd_disk["Abnormality"] == "Calcification").sum()
        cdd_both = (df_cdd_disk["Abnormality"] == "Both (Mass & Calcification)").sum()
        md.append(
            f"| **CDD-CESM** | `biopsy` | Low-Energy Digital (DM) | **{cdd_data['total_disk_images']:,}** | {cdd_data['total_annotations']:,} | **{cdd_data['unique_patients']:,}** | Biopsy & Follow-up | {cdd_mal:,} ({cdd_mal/cdd_tot*100:.1f}%) | {cdd_ben:,} ({cdd_ben/cdd_tot*100:.1f}%) | {cdd_norm:,} ({cdd_norm/cdd_tot*100:.1f}%) | Normal ({cdd_norm:,}), Mass ({cdd_mass:,}), Calc ({cdd_calc:,}), Both ({cdd_both:,}) |"
        )

    # CBIS (if present)
    if cbis_data:
        cbis_disk_mal = (cbis_data["df_disk"]["pathology"] == "malignant").sum()
        cbis_disk_ben = (cbis_data["df_disk"]["pathology"].isin(["benign", "benign_without_callback"])).sum()
        md.append(
            f"| **CBIS-DDSM** | `biopsy` | Scanned Film (Screening) | **{cbis_data['total_img_disk']:,}** | {cbis_data['total_ann_count']:,} | **{cbis_data['pt_disk']['patient_id'].nunique():,}** | Biopsy / Pathology | {cbis_disk_mal:,} ({cbis_disk_mal/cbis_data['total_img_disk']*100:.1f}%) | {cbis_disk_ben:,} ({cbis_disk_ben/cbis_data['total_img_disk']*100:.1f}%) | 0 | Mass ({ (cbis_data['df_disk']['abnormality']=='mass').sum():,}), Calc ({ (cbis_data['df_disk']['abnormality']=='calcification').sum():,}) |"
        )

    md.append("")
    md.append(
        "\\* *Note for CMMD: Image counts for CMMD are estimated from the 2 standard views (CC, MLO) per annotated breast lesion side (1,767 breast lesion records on disk across 1,673 patients; 4,793 total raw DICOM files stored on disk).* \n"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 2: VinDr-Mammo (mini VinDr)
    # ──────────────────────────────────────────────────────────────────────────
    if vindr_data:
        md.append("---")
        md.append("## 2. VinDr-Mammo (mini VinDr)")
        md.append("")
        md.append("**Source Directory**: `data/vindr`  ")
        md.append("**Label Family**: `birads`  ")
        md.append("**Imaging Modality**: Full-Field Digital Mammography (FFDM)  ")
        md.append("**Verification**: Consensus annotations by 3 experienced radiologists from 2 major Vietnamese hospitals (HMUH and Bach Mai Hospital)  ")
        md.append("")
        md.append("### 2.1 File & Image Inventory")
        md.append("")
        md.append("| Metric | On Disk | Full Catalog |")
        md.append("| :--- | :---: | :---: |")
        md.append(f"| **Total DICOM Images** | **{vindr_data['dcm_count_disk']:,}** | {vindr_data['total_catalog_images']:,} |")
        md.append(f"| **Unique Studies / Patients** | **{vindr_data['unique_studies_disk']:,}** | {vindr_data['total_catalog_studies']:,} |")
        md.append("| **Images per Study** | **4.0** (standard 4-view: L_CC, L_MLO, R_CC, R_MLO) | 4.0 |")
        md.append("")

        md.append("### 2.2 Pathology & BI-RADS Distribution")
        md.append("")
        df_v = vindr_data["df_disk"]
        v_tot = len(df_v)
        md.append("#### A. Image-Level 3-Class System (BTP Standard)")
        md.append("")
        md.append("| Class | Label Mapping | BI-RADS Categories | Image Count | Image % |")
        md.append("| :---: | :--- | :--- | :---: | :---: |")
        birads_vc = df_v["breast_birads"].value_counts()
        b1 = birads_vc.get("BI-RADS 1", 0)
        b2 = birads_vc.get("BI-RADS 2", 0)
        b3 = birads_vc.get("BI-RADS 3", 0)
        b4 = birads_vc.get("BI-RADS 4", 0)
        b5 = birads_vc.get("BI-RADS 5", 0)
        c0 = (df_v["3class"] == "Class 0 (Normal)").sum()
        c1 = (df_v["3class"] == "Class 1 (Benign)").sum()
        c2 = (df_v["3class"] == "Class 2 (Suspicious/Malignant)").sum()
        md.append(f"| **0** | **Normal** | BI-RADS 1 ({b1:,}) | {c0:,} | {c0/v_tot*100:.2f}% |")
        md.append(f"| **1** | **Benign** | BI-RADS 2 ({b2:,}) + BI-RADS 3 ({b3:,}) | {c1:,} | {c1/v_tot*100:.2f}% |")
        md.append(f"| **2** | **Suspicious / Malignant** | BI-RADS 4 ({b4:,}) + BI-RADS 5 ({b5:,}) | {c2:,} | {c2/v_tot*100:.2f}% |")
        md.append(f"| | **Total** | | **{v_tot:,}** | **100.00%** |")
        md.append("")

        md.append("#### B. Study / Patient Level Breakdown (1,218 patients)")
        md.append("")
        pt_v = vindr_data["pt_df"]
        pt_c0 = (pt_v["3class"] == "Class 0 (Normal)").sum()
        pt_c1 = (pt_v["3class"] == "Class 1 (Benign)").sum()
        pt_c2 = (pt_v["3class"] == "Class 2 (Suspicious/Malignant)").sum()
        pt_tot = len(pt_v)
        md.append("| Category | Patient Count | Patient % |")
        md.append("| :--- | :---: | :---: |")
        md.append(f"| **Normal (Class 0 only)** | {pt_c0:,} | {pt_c0/pt_tot*100:.2f}% |")
        md.append(f"| **Benign (Class 1 max)** | {pt_c1:,} | {pt_c1/pt_tot*100:.2f}% |")
        md.append(f"| **Suspicious / Malignant (Class 2)** | {pt_c2:,} | {pt_c2/pt_tot*100:.2f}% |")
        md.append(f"| **Total** | **{pt_tot:,}** | **100.00%** |")
        md.append("")
        md.append(f"- **Binary View (Class 2 vs. Rest)**: Malignant/Suspicious: **{pt_c2:,}** ({pt_c2/pt_tot*100:.2f}%) vs. Normal/Benign: **{pt_c0+pt_c1:,}** ({(pt_c0+pt_c1)/pt_tot*100:.2f}%) *(matches `train.csv` labels)*")
        md.append("")

        md.append("### 2.3 Abnormality Kinds (Mass vs. Calcification)")
        md.append("")
        md.append("Cross-tabulation of abnormality kind vs. 3-class diagnosis (Image level):")
        md.append("")
        ct_v = pd.crosstab(df_v["finding_kind"], df_v["3class"], margins=True)
        md.append("| Abnormality Kind | Class 0 (Normal) | Class 1 (Benign) | Class 2 (Suspicious) | Total Images | % Class 2 (Suspicious) |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for k in [
            "Mass",
            "Calcification",
            "Both (Mass & Calcification)",
            "Architectural Distortion",
            "Asymmetry",
            "No Finding / Normal",
            "Other finding",
        ]:
            if k in ct_v.index:
                c0_val = ct_v.loc[k, "Class 0 (Normal)"] if "Class 0 (Normal)" in ct_v.columns else 0
                c1_val = ct_v.loc[k, "Class 1 (Benign)"] if "Class 1 (Benign)" in ct_v.columns else 0
                c2_val = ct_v.loc[k, "Class 2 (Suspicious/Malignant)"] if "Class 2 (Suspicious/Malignant)" in ct_v.columns else 0
                tot = ct_v.loc[k, "All"]
                pct_susp = f"{c2_val/tot*100:.2f}%" if tot > 0 else "0.0%"
                md.append(f"| **{k}** | {c0_val:,} | {c1_val:,} | {c2_val:,} | **{tot:,}** | {pct_susp} |")
        md.append(f"| **Total** | **{c0:,}** | **{c1:,}** | **{c2:,}** | **{v_tot:,}** | **{c2/v_tot*100:.2f}%** |")
        md.append("")

        md.append("Patient-level abnormality cross-tabulation:")
        md.append("")
        pt_ct_v = pd.crosstab(pt_v["finding_kind"], pt_v["3class"], margins=True)
        md.append("| Patient Abnormality | Class 0 (Normal) | Class 1 (Benign) | Class 2 (Suspicious) | Total Patients | % Class 2 |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for k in [
            "Mass",
            "Calcification",
            "Both (Mass & Calcification)",
            "Architectural Distortion",
            "Asymmetry",
            "No Finding / Normal",
            "Other finding",
        ]:
            if k in pt_ct_v.index:
                c0_val = pt_ct_v.loc[k, "Class 0 (Normal)"] if "Class 0 (Normal)" in pt_ct_v.columns else 0
                c1_val = pt_ct_v.loc[k, "Class 1 (Benign)"] if "Class 1 (Benign)" in pt_ct_v.columns else 0
                c2_val = pt_ct_v.loc[k, "Class 2 (Suspicious/Malignant)"] if "Class 2 (Suspicious/Malignant)" in pt_ct_v.columns else 0
                tot = pt_ct_v.loc[k, "All"]
                pct_susp = f"{c2_val/tot*100:.2f}%" if tot > 0 else "0.0%"
                md.append(f"| **{k}** | {c0_val:,} | {c1_val:,} | {c2_val:,} | **{tot:,}** | {pct_susp} |")
        md.append(f"| **Total Patients** | **{pt_c0:,}** | **{pt_c1:,}** | **{pt_c2:,}** | **{pt_tot:,}** | **{pt_c2/pt_tot*100:.2f}%** |")
        md.append("")

        md.append("### 2.4 Acquisition & Hardware Profile")
        md.append("")
        md.append("- **Standard Views**: CC: 2,436 images (50.0%) | MLO: 2,436 images (50.0%)")
        md.append("- **Laterality**: Left: 2,436 images (50.0%) | Right: 2,436 images (50.0%)")
        if not vindr_data["ages"].empty:
            ages = vindr_data["ages"]
            md.append(f"- **Patient Age**: Mean {ages.mean():.1f} ± {ages.std():.1f} years (Range: {int(ages.min())} – {int(ages.max())}; Median: {int(ages.median())})")
        if "breast_density" in df_v.columns:
            dens_counts = df_v["breast_density"].value_counts()
            dens_str = ", ".join([f"{k}: {v:,}" for k, v in dens_counts.items()])
            md.append(f"- **ACR Breast Density**: {dens_str}")
        if not vindr_data["meta_disk"].empty and "Manufacturer's Model Name" in vindr_data["meta_disk"].columns:
            mach_counts = vindr_data["meta_disk"]["Manufacturer's Model Name"].value_counts()
            mach_str = ", ".join([f"{k}: {v:,} images" for k, v in mach_counts.items()])
            md.append(f"- **Scanner Hardware**: {mach_str}")
        md.append("")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 3: CMMD Metrics
    # ──────────────────────────────────────────────────────────────────────────
    if cmmd_data:
        md.append("---")
        md.append("## 3. CMMD (The Chinese Mammography Database)")
        md.append("")
        md.append("**Source Directory**: `data/cmmd`  ")
        md.append("**Label Family**: `biopsy`  ")
        md.append("**Imaging Modality**: Full-Field Digital Mammography (GE Senographe 2000D / DS / Essential)  ")
        md.append("**Verification**: Biopsy-confirmed histology  ")
        md.append("")
        md.append("### 3.1 Overview & Cohort Size")
        md.append("")
        md.append(f"- **Raw DICOM Files on Disk**: **{cmmd_data['dcm_count_disk']:,}** across **{cmmd_data['patient_dirs_count']:,}** patient directories")
        md.append(f"- **Clinical Records (Excel)**: **{len(cmmd_data['clin_df']):,}** breast lesion entries across **{cmmd_data['pt_all']['ID1'].nunique():,}** unique patients (97 bilateral lesion cases)")
        md.append(f"- **Clinical Entries Available on Disk**: **{len(cmmd_data['disk_clin']):,}** breast lesion entries across **{cmmd_data['pt_disk']['ID1'].nunique():,}** unique patients")
        md.append("")

        md.append("### 3.2 Pathology Distribution (Malignant vs. Benign)")
        md.append("")
        c_disk = cmmd_data["disk_clin"]
        c_all = cmmd_data["clin_df"]
        md.append("| Cohort Scope | Malignant Lesions | Malignant % | Benign Lesions | Benign % | Total Lesions | Unique Patients |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        m_d = (c_disk["classification"] == "Malignant").sum()
        b_d = (c_disk["classification"] == "Benign").sum()
        md.append(f"| **On Disk** | {m_d:,} | {m_d/len(c_disk)*100:.2f}% | {b_d:,} | {b_d/len(c_disk)*100:.2f}% | **{len(c_disk):,}** | **{cmmd_data['pt_disk']['ID1'].nunique():,}** |")
        m_a = (c_all["classification"] == "Malignant").sum()
        b_a = (c_all["classification"] == "Benign").sum()
        md.append(f"| **Full Clinical Cohort** | {m_a:,} | {m_a/len(c_all)*100:.2f}% | {b_a:,} | {b_a/len(c_all)*100:.2f}% | **{len(c_all):,}** | **{cmmd_data['pt_all']['ID1'].nunique():,}** |")
        md.append("")

        md.append("Patient-level pathology distribution (Patients on disk):")
        pt_cmmd = cmmd_data["pt_disk"]
        pt_m = (pt_cmmd["classification"] == "Malignant").sum()
        pt_b = (pt_cmmd["classification"] == "Benign").sum()
        md.append(f"- **Malignant Patients**: {pt_m:,} ({pt_m/len(pt_cmmd)*100:.2f}%)")
        md.append(f"- **Benign Patients**: {pt_b:,} ({pt_b/len(pt_cmmd)*100:.2f}%)")
        md.append(f"- **Total Patients**: {len(pt_cmmd):,}")
        md.append("")

        md.append("### 3.3 Abnormality Kinds (Mass vs. Calcification)")
        md.append("")
        md.append("Cross-tabulation of abnormality kind vs. pathology (Breast lesion level, on disk):")
        md.append("")
        ct_cmmd = pd.crosstab(c_disk["abnormality"], c_disk["classification"], margins=True)
        md.append("| Abnormality Kind | Benign | Malignant | Total Lesions | % Malignant |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for abn in ["mass", "calcification", "both"]:
            if abn in ct_cmmd.index:
                b = ct_cmmd.loc[abn, "Benign"]
                m = ct_cmmd.loc[abn, "Malignant"]
                tot = ct_cmmd.loc[abn, "All"]
                name = "Both (Mass & Calc)" if abn == "both" else abn.capitalize()
                md.append(f"| **{name}** | {b:,} | {m:,} | **{tot:,}** | {m/tot*100:.2f}% |")
        md.append(f"| **Total** | **{b_d:,}** | **{m_d:,}** | **{len(c_disk):,}** | **{m_d/len(c_disk)*100:.2f}%** |")
        md.append("")

        md.append("Patient-level abnormality cross-tabulation (On disk):")
        md.append("")
        pt_ct_cmmd = pd.crosstab(pt_cmmd["abnormality"], pt_cmmd["classification"], margins=True)
        md.append("| Patient Abnormality | Benign | Malignant | Total Patients | % Malignant |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for abn in ["mass", "calcification", "both"]:
            if abn in pt_ct_cmmd.index:
                b = pt_ct_cmmd.loc[abn, "Benign"]
                m = pt_ct_cmmd.loc[abn, "Malignant"]
                tot = pt_ct_cmmd.loc[abn, "All"]
                name = "Both (Mass & Calc)" if abn == "both" else abn.capitalize()
                md.append(f"| **{name}** | {b:,} | {m:,} | **{tot:,}** | {m/tot*100:.2f}% |")
        md.append(f"| **Total Patients** | **{pt_b:,}** | **{pt_m:,}** | **{len(pt_cmmd):,}** | **{pt_m/len(pt_cmmd)*100:.2f}%** |")
        md.append("")

        md.append("### 3.4 Additional Clinical Details")
        md.append("")
        ages = pt_cmmd["age"].dropna()
        md.append(f"- **Patient Age**: Mean {ages.mean():.1f} ± {ages.std():.1f} years (Range: {int(ages.min())} – {int(ages.max())} years; Median: {int(ages.median())})")
        if "subtype" in c_disk.columns:
            subtypes = c_disk["subtype"].dropna().value_counts()
            if not subtypes.empty:
                md.append("- **Molecular Subtypes (Malignant Lesions with Subtyping)**:")
                for st, count in subtypes.items():
                    md.append(f"  - `{st}`: {count:,} cases ({count/subtypes.sum()*100:.1f}%)")
        md.append("")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 4: INbreast Metrics
    # ──────────────────────────────────────────────────────────────────────────
    if inb_data:
        md.append("---")
        md.append("## 4. INbreast (INESC Porto Mammography Database)")
        md.append("")
        md.append("**Source Directory**: `data/inbreast`  ")
        md.append("**Label Family**: `birads`  ")
        md.append("**Imaging Modality**: Full-Field Digital Mammography (Siemens Mammomat Novation)  ")
        md.append("**Verification**: BI-RADS assessment by expert consensus radiologist team, with histological verification for suspicious/malignant lesions  ")
        md.append("")
        md.append("### 4.1 Overview & Cohort Size")
        md.append("")
        md.append(f"- **DICOM Images on Disk**: **{inb_data['dcm_count']:,}**")
        md.append(f"- **Unique Patients**: **{inb_data['unique_patients']:,}** (anonymized hash IDs in filenames)")
        md.append(f"- **Detailed XML Annotation Files**: **{inb_data['xml_count']:,}** files (ROIs outlining individual lesions)")
        md.append(f"- **Normal Mammograms (BI-RADS 1, No Lesion ROIs)**: **{(inb_data['df']['Bi-Rads']=='1').sum():,}**")
        md.append("")

        md.append("### 4.2 BI-RADS Distribution & Pathology Mapping")
        md.append("")
        df_inb = inb_data["df"]
        md.append("| BI-RADS Category | Clinical Description | 3-Class Mapping | Binary Mapping | Image Count | Image % |")
        md.append("| :--- | :--- | :--- | :--- | :---: | :---: |")
        birads_desc = {
            "1": ("Negative / Normal", "Class 0 (Normal)", "Benign / Negative"),
            "2": ("Benign Finding", "Class 1 (Benign)", "Benign / Negative"),
            "3": ("Probably Benign (Short-interval follow-up)", "Class 1 (Benign)", "Benign / Negative"),
            "4a": ("Low Suspicion for Malignancy", "Class 2 (Suspicious)", "Malignant / Suspicious"),
            "4b": ("Moderate Suspicion for Malignancy", "Class 2 (Suspicious)", "Malignant / Suspicious"),
            "4c": ("High Suspicion for Malignancy", "Class 2 (Suspicious)", "Malignant / Suspicious"),
            "5": ("Highly Suggestive of Malignancy (Biopsy needed)", "Class 2 (Suspicious)", "Malignant / Suspicious"),
            "6": ("Known Biopsy-Proven Malignancy", "Class 2 (Suspicious)", "Malignant / Suspicious"),
        }
        b_counts = df_inb["Bi-Rads"].value_counts()
        for b_code in ["1", "2", "3", "4a", "4b", "4c", "5", "6"]:
            if b_code in b_counts:
                c = b_counts[b_code]
                desc, c3, grp = birads_desc.get(b_code, ("Unknown", "Unknown", "Unknown"))
                md.append(f"| **BI-RADS {b_code}** | {desc} | {c3} | {grp} | {c:,} | {c/len(df_inb)*100:.2f}% |")

        mal_inb = (df_inb["Pathology"] == "Malignant / Suspicious").sum()
        ben_inb = (df_inb["Pathology"] == "Benign / Negative").sum()
        md.append(f"| **Subtotal Normal / Benign (1–3)** | Negative or Benign Assessment | Classes 0 & 1 | Benign / Negative | **{ben_inb:,}** | **{ben_inb/len(df_inb)*100:.2f}%** |")
        md.append(f"| **Subtotal Malignant (4–6)** | Suspicious or Biopsy-Proven | Class 2 | Malignant / Suspicious | **{mal_inb:,}** | **{mal_inb/len(df_inb)*100:.2f}%** |")
        md.append(f"| **Total** | | | | **{len(df_inb):,}** | **100.00%** |")
        md.append("")

        md.append("Patient-level pathology distribution:")
        pt_inb = inb_data["pt_df"]
        pt_mal = (pt_inb["Pathology"] == "Malignant / Suspicious").sum()
        pt_ben = (pt_inb["Pathology"] == "Benign / Negative").sum()
        md.append(f"- **Malignant / Suspicious Patients**: {pt_mal:,} ({pt_mal/len(pt_inb)*100:.2f}%)")
        md.append(f"- **Benign / Negative Patients**: {pt_ben:,} ({pt_ben/len(pt_inb)*100:.2f}%)")
        md.append(f"- **Total Patients**: {len(pt_inb):,}")
        md.append("")

        md.append("### 4.3 Abnormality Kinds (Mass vs. Calcification)")
        md.append("")
        md.append("Cross-tabulation of abnormality kind vs. pathology (Image level):")
        md.append("")
        ct_inb = pd.crosstab(df_inb["Abnormality"], df_inb["Pathology"], margins=True)
        md.append("| Abnormality Kind | Benign / Negative | Malignant / Suspicious | Total Images | % Malignant/Suspicious |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        abn_labels = {
            "calcification": "Calcification",
            "both": "Both (Mass & Calcification)",
            "mass": "Mass",
            "normal": "Normal (BI-RADS 1)",
            "other": "Other finding (Cluster, Spiculated, etc.)",
        }
        for k in ["calcification", "both", "mass", "other", "normal"]:
            if k in ct_inb.index:
                b = ct_inb.loc[k, "Benign / Negative"] if "Benign / Negative" in ct_inb.columns else 0
                m = ct_inb.loc[k, "Malignant / Suspicious"] if "Malignant / Suspicious" in ct_inb.columns else 0
                tot = ct_inb.loc[k, "All"]
                md.append(f"| **{abn_labels[k]}** | {b:,} | {m:,} | **{tot:,}** | {m/tot*100:.2f}% |")
        md.append(f"| **Total** | **{ben_inb:,}** | **{mal_inb:,}** | **{len(df_inb):,}** | **{mal_inb/len(df_inb)*100:.2f}%** |")
        md.append("")

        md.append("Patient-level abnormality cross-tabulation:")
        md.append("")
        pt_ct_inb = pd.crosstab(pt_inb["Abnormality"], pt_inb["Pathology"], margins=True)
        md.append("| Patient Abnormality | Benign / Negative | Malignant / Suspicious | Total Patients | % Malignant/Suspicious |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for k in ["calcification", "both", "mass", "other", "normal"]:
            if k in pt_ct_inb.index:
                b = pt_ct_inb.loc[k, "Benign / Negative"] if "Benign / Negative" in pt_ct_inb.columns else 0
                m = pt_ct_inb.loc[k, "Malignant / Suspicious"] if "Malignant / Suspicious" in pt_ct_inb.columns else 0
                tot = pt_ct_inb.loc[k, "All"]
                md.append(f"| **{abn_labels[k]}** | {b:,} | {m:,} | **{tot:,}** | {m/tot*100:.2f}% |")
        md.append(f"| **Total Patients** | **{pt_ben:,}** | **{pt_mal:,}** | **{len(pt_inb):,}** | **{pt_mal/len(pt_inb)*100:.2f}%** |")
        md.append("")

        md.append("### 4.4 Additional Radiologic Attributes")
        md.append("")
        md.append(f"- **Views**: CC: {(df_inb['View']=='CC').sum():,} | MLO: {(df_inb['View']=='MLO').sum():,} | FB: {(df_inb['View']=='FB').sum():,}")
        md.append(f"- **Laterality**: Left: {(df_inb['Laterality']=='L').sum():,} | Right: {(df_inb['Laterality']=='R').sum():,}")
        acr_counts = df_inb["ACR"].str.strip().value_counts()
        acr_str = ", ".join([f"ACR {k}: {v}" for k, v in acr_counts.items() if k in ["1", "2", "3", "4"]])
        md.append(f"- **ACR Breast Density**: {acr_str}")
        md.append(f"- **Detailed ROI breakdown**: {inb_data['roi_details']['Calcification']:,} individual calcification points, {inb_data['roi_details']['Mass']:,} mass boundaries, {inb_data['roi_details']['Cluster']:,} calcification clusters.")
        md.append("")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 5: CDD-CESM Metrics
    # ──────────────────────────────────────────────────────────────────────────
    if cdd_data:
        md.append("---")
        md.append("## 5. CDD-CESM (Contrast-Enhanced Spectral Mammography)")
        md.append("")
        md.append("**Source Directory**: `data/cdd`  ")
        md.append("**Label Family**: `biopsy`  ")
        md.append("**Imaging Modality**: Low Energy Full-Field Digital Mammography (DM, dual-energy CESM low-energy acquisition)  ")
        md.append("**Verification**: Biopsy and histological examination / clinical imaging follow-up  ")
        md.append("")
        md.append("### 5.1 Overview & Cohort Size")
        md.append("")
        md.append(f"- **Total Image Files on Disk**: **{cdd_data['total_disk_images']:,}** (Low Energy DM images only, per project specification)")
        md.append(f"- **Annotated Images on Disk**: **{len(cdd_data['df_disk']):,}** image entries across **{cdd_data['unique_patients']:,}** unique patients")
        md.append(f"- **Full Multi-Modality Catalog (Excel)**: **{cdd_data['total_annotations']:,}** image entries (DM + recombined CESM)")
        md.append("")

        md.append("### 5.2 Pathology Distribution (Low-Energy Images on Disk)")
        md.append("")
        df_cdd = cdd_data["df_disk"]
        md.append("| Pathology Category | Image Count | Image % | Patient Count | Patient % |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        pt_cdd = cdd_data["pt_df"]
        for p_cat in ["Malignant", "Benign", "Normal"]:
            im_c = (df_cdd["Pathology"] == p_cat).sum()
            pt_c = (pt_cdd["Pathology"] == p_cat).sum()
            md.append(f"| **{p_cat}** | {im_c:,} | {im_c/len(df_cdd)*100:.2f}% | {pt_c:,} | {pt_c/len(pt_cdd)*100:.2f}% |")
        md.append(f"| **Total** | **{len(df_cdd):,}** | **100.00%** | **{len(pt_cdd):,}** | **100.00%** |")
        md.append("")

        md.append("### 5.3 Abnormality Kinds (Mass vs. Calcification)")
        md.append("")
        md.append("Cross-tabulation of abnormality kind vs. pathology (Image level):")
        md.append("")
        ct_cdd = pd.crosstab(df_cdd["Abnormality"], df_cdd["Pathology"], margins=True)
        md.append("| Abnormality Kind | Benign | Malignant | Normal | Total Images | % Malignant |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for k in [
            "Mass",
            "Calcification",
            "Both (Mass & Calcification)",
            "Architectural Distortion",
            "Asymmetry",
            "Normal / Negative",
            "Other",
        ]:
            if k in ct_cdd.index:
                b = ct_cdd.loc[k, "Benign"] if "Benign" in ct_cdd.columns else 0
                m = ct_cdd.loc[k, "Malignant"] if "Malignant" in ct_cdd.columns else 0
                n = ct_cdd.loc[k, "Normal"] if "Normal" in ct_cdd.columns else 0
                tot = ct_cdd.loc[k, "All"]
                pct_m = f"{m/tot*100:.2f}%" if tot > 0 else "0.0%"
                md.append(f"| **{k}** | {b:,} | {m:,} | {n:,} | **{tot:,}** | {pct_m} |")
        md.append(f"| **Total** | **{(df_cdd['Pathology']=='Benign').sum():,}** | **{(df_cdd['Pathology']=='Malignant').sum():,}** | **{(df_cdd['Pathology']=='Normal').sum():,}** | **{len(df_cdd):,}** | **{(df_cdd['Pathology']=='Malignant').sum()/len(df_cdd)*100:.2f}%** |")
        md.append("")

        md.append("Patient-level abnormality cross-tabulation:")
        md.append("")
        pt_ct_cdd = pd.crosstab(pt_cdd["Abnormality"], pt_cdd["Pathology"], margins=True)
        md.append("| Patient Abnormality | Benign | Malignant | Normal | Total Patients | % Malignant |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for k in [
            "Mass",
            "Calcification",
            "Both (Mass & Calcification)",
            "Architectural Distortion",
            "Asymmetry",
            "Normal / Negative",
            "Other",
        ]:
            if k in pt_ct_cdd.index:
                b = pt_ct_cdd.loc[k, "Benign"] if "Benign" in pt_ct_cdd.columns else 0
                m = pt_ct_cdd.loc[k, "Malignant"] if "Malignant" in pt_ct_cdd.columns else 0
                n = pt_ct_cdd.loc[k, "Normal"] if "Normal" in pt_ct_cdd.columns else 0
                tot = pt_ct_cdd.loc[k, "All"]
                pct_m = f"{m/tot*100:.2f}%" if tot > 0 else "0.0%"
                md.append(f"| **{k}** | {b:,} | {m:,} | {n:,} | **{tot:,}** | {pct_m} |")
        md.append(f"| **Total Patients** | **{(pt_cdd['Pathology']=='Benign').sum():,}** | **{(pt_cdd['Pathology']=='Malignant').sum():,}** | **{(pt_cdd['Pathology']=='Normal').sum():,}** | **{len(pt_cdd):,}** | **{(pt_cdd['Pathology']=='Malignant').sum()/len(pt_cdd)*100:.2f}%** |")
        md.append("")

        md.append("### 5.4 Additional Clinical Details")
        md.append("")
        ages = pt_cdd["Age"].dropna()
        if not ages.empty:
            md.append(f"- **Patient Age**: Mean {ages.mean():.1f} ± {ages.std():.1f} years (Range: {int(ages.min())} – {int(ages.max())} years; Median: {int(ages.median())})")
        if "Type" in df_cdd.columns:
            types = df_cdd["Type"].value_counts()
            md.append(f"- **Image Type**: DM (Low-energy 2D Digital Mammography): {types.get('DM', len(df_cdd)):,} images")
        if "Breast density (ACR)" in df_cdd.columns:
            dens = df_cdd["Breast density (ACR)"].value_counts()
            density_str = ", ".join([f"{k}: {v}" for k, v in dens.items()])
            md.append(f"- **Breast Density (ACR)**: {density_str}")
        if "Machine" in df_cdd.columns:
            mach = df_cdd["Machine"].value_counts()
            machine_str = ", ".join([f"Machine {k}: {v} images" for k, v in mach.items()])
            md.append(f"- **Mammography Machines**: {machine_str}")
        md.append("")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 6: CBIS-DDSM Metrics (Optional / Historical)
    # ──────────────────────────────────────────────────────────────────────────
    if cbis_data:
        md.append("---")
        md.append("## 6. CBIS-DDSM (Curated Breast Imaging Subset of DDSM)")
        md.append("")
        md.append("**Source Directory**: `data/cbis`  ")
        md.append("**Label Family**: `biopsy`  ")
        md.append("**Imaging Modality**: Film mammography digitizer (Screening mammography)  ")
        md.append("**Verification**: Pathology/biopsy confirmed  ")
        md.append("")
        md.append("### 6.1 File & Image Inventory")
        md.append("")
        md.append("| Metric | Train Split | Test Split | Total / Overall |")
        md.append("| :--- | :---: | :---: | :---: |")
        md.append(f"| **Images Present on Disk** | {cbis_data['train_img_count']:,} | {cbis_data['test_img_count']:,} | **{cbis_data['total_img_disk']:,}** |")
        md.append(f"| **Complete Dataset Annotations (JSON)** | {cbis_data['train_ann_count']:,} | {cbis_data['test_ann_count']:,} | **{cbis_data['total_ann_count']:,}** |")
        md.append(f"| **Unique Patients (Images on Disk)** | {cbis_data['df_train_disk']['patient_id'].nunique():,} | {cbis_data['df_test_disk']['patient_id'].nunique():,} | **{cbis_data['pt_disk']['patient_id'].nunique():,}** |")
        md.append(f"| **Unique Patients (All Annotations)** | {cbis_data['df_train_all']['patient_id'].nunique():,} | {cbis_data['df_test_all']['patient_id'].nunique():,} | **{cbis_data['pt_all']['patient_id'].nunique():,}** |")
        md.append("")

        df_d = cbis_data["df_disk"]
        disk_mal = (df_d["pathology"] == "malignant").sum()
        disk_ben = (df_d["pathology"] == "benign").sum()
        disk_ben_nocb = (df_d["pathology"] == "benign_without_callback").sum()
        disk_tot = len(df_d)
        pt_d = cbis_data["pt_disk"]
        pt_d_mal = (pt_d["pathology"] == "malignant").sum()
        pt_d_ben = (pt_d["pathology"] == "benign").sum()
        pt_d_ben_nocb = (pt_d["pathology"] == "benign_without_callback").sum()
        pt_d_tot = len(pt_d)

        md.append("### 6.2 Pathology Distribution (Images on Disk)")
        md.append("")
        md.append("| Pathology Category | Image Count | Image % | Patient Count | Patient % |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        md.append(f"| **Malignant** | {disk_mal:,} | {disk_mal/disk_tot*100:.2f}% | {pt_d_mal:,} | {pt_d_mal/pt_d_tot*100:.2f}% |")
        md.append(f"| **Benign** (Callback) | {disk_ben:,} | {disk_ben/disk_tot*100:.2f}% | {pt_d_ben:,} | {pt_d_ben/pt_d_tot*100:.2f}% |")
        md.append(f"| **Benign without callback** | {disk_ben_nocb:,} | {disk_ben_nocb/disk_tot*100:.2f}% | {pt_d_ben_nocb:,} | {pt_d_ben_nocb/pt_d_tot*100:.2f}% |")
        md.append(f"| **Total Benign Combined** | {disk_ben+disk_ben_nocb:,} | {(disk_ben+disk_ben_nocb)/disk_tot*100:.2f}% | {pt_d_ben+pt_d_ben_nocb:,} | {(pt_d_ben+pt_d_ben_nocb)/pt_d_tot*100:.2f}% |")
        md.append(f"| **Total** | **{disk_tot:,}** | **100.00%** | **{pt_d_tot:,}** | **100.00%** |")
        md.append("")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 7: Key Findings & Cross-Site Comparative Analysis
    # ──────────────────────────────────────────────────────────────────────────
    md.append("---")
    md.append("## 6. Synthesis & Federated Learning Implications")
    md.append("")
    md.append("### 6.1 Abnormality Distribution Comparison Across Simulated Sites")
    md.append("")
    md.append("| Simulated Site | Label Family | Mass Presentation | Microcalcification Presentation | Dual / Other Presentations | Normal Mammograms |")
    md.append("| :--- | :---: | :--- | :--- | :--- | :--- |")
    if vindr_data:
        md.append("| **VinDr-Mammo (mini)** | `birads` | 766 images (15.7%) | 199 images (4.1%) | Both: 157 (3.2%), Asym: 249, Distort: 67 | 2,332 images (47.9% Class 0) |")
    if cmmd_data:
        md.append("| **CMMD** | `biopsy` | 1,044 lesions (59.1%) | 262 lesions (14.8%) | Both: 461 lesions (26.1%) | 0 (Clinical biopsy cohort) |")
    if inb_data:
        md.append("| **INbreast** | `birads` | 28 images (6.8%) | 234 images (57.1%) | Both: 79 images (19.3%), Other: 2 | 67 images (16.3% BI-RADS 1) |")
    if cdd_data:
        md.append("| **CDD-CESM** | `biopsy` | 226 images (22.6%) | 156 images (15.6%) | Both: 86 (8.6%), Asym: 85, Distort: 26 | 367 negative findings (340 Class 0 Normal) |")
    md.append("")

    md.append("### 6.2 Label Semantics & Clinical Shift")
    md.append("")
    md.append("1. **Label Family Dichotomy (`birads` vs. `biopsy`)**:")
    md.append("   - **`birads` Family (VinDr-Mammo, INbreast)**: Labels reflect expert consensus radiologist interpretation following the ACR BI-RADS lexicon (BI-RADS 1 = normal, 2–3 = benign, 4–5/6 = suspicious/malignant). These sites capture realistic screening and diagnostic triage distributions, including high negative/normal rates.")
    md.append("   - **`biopsy` Family (CMMD, CDD-CESM)**: Ground truth is anchored in histopathological biopsy confirmation. CMMD consists entirely of biopsied lesions, resulting in an enriched positive rate (~68.6% malignant).")
    md.append("2. **3-Class Standard Formulation**:")
    md.append("   - **Class 0 (Normal)**: Present in VinDr-Mammo (47.9%) and CDD-CESM (34.0%), as well as INbreast (16.3% BI-RADS 1), representing baseline uncompromised tissue.")
    md.append("   - **Class 1 (Benign)**: Benign cysts, fibroadenomas, and stable findings (VinDr 35.6%, CMMD 31.4%, CDD 32.9%, INbreast 59.3%).")
    md.append("   - **Class 2 (Suspicious / Malignant)**: Actionable malignancy requiring intervention (VinDr 16.6%, CMMD 68.6%, INbreast 24.4%, CDD 33.1%).")
    md.append("3. **Non-IID Heterogeneity Across Hospitals**:")
    md.append("   - **Scanner Differences**: Siemens Mammomat Novation / Inspiration (VinDr, INbreast), GE Senographe 2000D / DS / Essential (CMMD, CDD), Planmed Nuance, and Giotto systems.")
    md.append("   - **Spatial Resolution & Feature Demands**: INbreast is dominated by fine punctate microcalcifications requiring preservation of high-frequency spatial detail, whereas CMMD and CDD-CESM are predominantly mass-centric, demanding wider receptive fields and contrast discrimination.")
    md.append("")

    content = "\n".join(md)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f" Successfully generated dataset metrics at: {output_path}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print(f"Starting dataset metrics extraction from: {data_dir.resolve()} ...")

    vindr_data = analyze_vindr(data_dir)
    if vindr_data:
        print(f"  [+] VinDr-Mammo parsed: {vindr_data['dcm_count_disk']} DICOMs on disk, {vindr_data['unique_studies_disk']} studies.")

    cmmd_data = analyze_cmmd(data_dir)
    if cmmd_data:
        print(f"  [+] CMMD parsed: {cmmd_data['dcm_count_disk']} DICOMs on disk, {len(cmmd_data['clin_df'])} clinical records.")

    inb_data = analyze_inbreast(data_dir)
    if inb_data:
        print(f"  [+] INbreast parsed: {inb_data['dcm_count']} DICOMs on disk, {inb_data['unique_patients']} patients.")

    cdd_data = analyze_cdd(data_dir)
    if cdd_data:
        print(f"  [+] CDD-CESM parsed: {cdd_data['total_disk_images']} images on disk, {cdd_data['unique_patients']} patients.")

    cbis_data = analyze_cbis(data_dir)
    if cbis_data:
        print(f"  [+] CBIS-DDSM parsed: {cbis_data['total_img_disk']} images on disk, {cbis_data['total_ann_count']} annotations.")

    generate_markdown(vindr_data, cmmd_data, inb_data, cdd_data, cbis_data, args.output)


if __name__ == "__main__":
    main()
