#!/usr/bin/env python3
"""Join CMMD meta.csv (from ingest_dicom.py) with the CMMD clinical spreadsheet -> labels.csv.

    python scripts/build_labels_cmmd.py --clinical data/cmmd/CMMD_clinicaldata_revision.xlsx

Column names are detected case-insensitively (expects something like ID1, [ID2], LeftRight,
classification with values Benign/Malignant). If detection fails the script prints the columns it
found, so adjust the candidate lists below.

Notes
- TCIA stores ID2 (when present) as the DICOM PatientID, otherwise ID1; the script mirrors that.
- Labels attach to (patient, breast side). Images of a breast with no clinical row are dropped.
- If a (patient, side) has several rows with different labels, malignant wins.
"""
import argparse
from pathlib import Path

import pandas as pd

ID1 = ("id1",)
ID2 = ("id2",)
SIDE = ("leftright", "lr", "side", "laterality")
CLS = ("classification", "class", "label", "diagnosis")


def norm(c):
    return str(c).strip().lower().replace(" ", "").replace("_", "")


def pick(cols, cands, what, required=True):
    for c in cands:
        if c in cols:
            return c
    if required:
        raise SystemExit(f"Could not find a '{what}' column. Columns found: {list(cols)}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--site-dir", default="data/cmmd")
    a = ap.parse_args()

    d = Path(a.site_dir)
    meta = pd.read_csv(d / "meta.csv", dtype=str).fillna("")
    cp = Path(a.clinical)
    clin = pd.read_excel(cp) if cp.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(cp)
    clin.columns = [norm(c) for c in clin.columns]
    cols = list(clin.columns)

    c_id1, c_id2 = pick(cols, ID1, "ID1"), pick(cols, ID2, "ID2", required=False)
    c_side, c_cls = pick(cols, SIDE, "left/right", required=False), pick(cols, CLS, "classification")
    print(f"using columns: id1={c_id1} id2={c_id2} side={c_side} class={c_cls}")

    pid = clin[c_id1].astype(str).str.strip()
    if c_id2:
        id2 = clin[c_id2].astype(str).str.strip()
        has2 = clin[c_id2].notna() & (id2 != "") & (id2.str.lower() != "nan")
        pid = pid.where(~has2, id2)
    clin["patient_id"] = pid
    clin["label"] = clin[c_cls].astype(str).str.strip().str.lower().map({"benign": 0, "malignant": 1})
    bad = clin["label"].isna().sum()
    if bad:
        print(f"note: {bad} clinical rows have a classification other than Benign/Malignant and are ignored")
    clin = clin.dropna(subset=["label"])

    meta["laterality"] = meta["laterality"].str.upper().str[:1]
    match_col = "original_patient_id" if "original_patient_id" in meta.columns else "patient_id"
    join_keys = [match_col]
    if c_side:
        clin["laterality"] = clin[c_side].astype(str).str.strip().str.upper().str[:1]
        join_keys.append("laterality")

    lab_keys = ["patient_id"] + (["laterality"] if c_side else [])
    lab = clin.groupby(lab_keys, as_index=False)["label"].max()
    if match_col != "patient_id":
        lab = lab.rename(columns={"patient_id": match_col})

    out = meta.merge(lab, on=join_keys, how="left")
    matched = out["label"].notna().sum()
    if matched == 0:
        print("No images matched. Example image PatientIDs:", meta[match_col].head(3).tolist())
        print("Example clinical IDs:", clin["patient_id"].head(3).tolist())
        raise SystemExit("ID formats do not line up; check ID1/ID2 handling.")
    print(f"{matched}/{len(out)} images matched a clinical row; dropping {len(out) - matched} unmatched")
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)
    out["site"] = "cmmd"
    out["category"] = ""
    out["raw_path"] = "raw512/" + out["sop_uid"] + ".png"
    out["proc_path"] = "proc512/" + out["sop_uid"] + ".png"
    keep = ["sop_uid", "patient_id", "laterality", "view", "label", "category", "site",
            "raw_path", "proc_path"]
    if "original_patient_id" in out.columns:
        keep.insert(2, "original_patient_id")
    for extra in ["file_name", "src_path", "manufacturer", "model"]:
        if extra in out.columns:
            keep.append(extra)
    out[keep].to_csv(d / "labels.csv", index=False)
    print(f"wrote {d/'labels.csv'}: {len(out)} images, {out['patient_id'].nunique()} patients, "
          f"label counts {out['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
