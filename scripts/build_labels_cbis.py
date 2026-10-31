#!/usr/bin/env python3
"""Build CBIS labels.csv from meta.csv and CBIS-DDSM case description CSVs.

Usage:
    python scripts/build_labels_cbis.py
    python scripts/build_labels_cbis.py --site-dir data/CBIS --csv-dir data/CBIS/csv

Matches each image by filename pattern or patient+laterality+view tuple.
Pathology mapping:
    BENIGN -> 0
    BENIGN_WITHOUT_CALLBACK -> 0
    MALIGNANT -> 1
If an image has multiple abnormalities, MALIGNANT (1) takes precedence.
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def load_cbis_cases(csv_dir):
    dfs = []
    p_dir = Path(csv_dir)
    for p in sorted(p_dir.glob("*_case_description_*.csv")):
        if "UPDATED" in p.name.upper():
            continue
        df = pd.read_csv(p, dtype=str).fillna("")
        df.columns = [c.strip().lower() for c in df.columns]
        dfs.append(df)
    if not dfs:
        raise SystemExit(f"No case description CSVs found in {csv_dir}")
    return pd.concat(dfs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Build CBIS labels.csv from meta.csv and case descriptions")
    ap.add_argument("--site-dir", default="data/CBIS",
                    help="directory containing meta.csv / where labels.csv will be saved (default: data/CBIS)")
    ap.add_argument("--csv-dir", default="data/CBIS/csv",
                    help="directory containing case description CSVs (default: data/CBIS/csv)")
    a = ap.parse_args()

    site_dir = Path(a.site_dir)
    meta_path = site_dir / "meta.csv"
    if not meta_path.exists():
        if (site_dir / "labels.csv").exists():
            meta_path = site_dir / "labels.csv"
        else:
            raise SystemExit(f"Neither meta.csv nor labels.csv found in {site_dir}. Run ingest_png.py first.")

    meta = pd.read_csv(meta_path, dtype=str).fillna("")
    if "sop_uid" not in meta.columns:
        raise SystemExit(f"Expected 'sop_uid' in {meta_path}")

    cases = load_cbis_cases(a.csv_dir)

    label_map = {
        "benign": 0,
        "benign_without_callback": 0,
        "malignant": 1,
    }
    cases["label_num"] = cases["pathology"].astype(str).str.strip().str.lower().map(label_map)
    valid_cases = cases.dropna(subset=["label_num"]).copy()
    valid_cases["label_num"] = valid_cases["label_num"].astype(int)

    # 1. Match by prefix from 'image file path'
    # Format e.g.: Calc-Training_P_00005_RIGHT_CC/1.3.6.1...
    prefix_labels = {}
    for _, r in valid_cases.iterrows():
        raw_fp = str(r.get("image file path", "")).strip()
        pfx = raw_fp.split("/")[0].strip()
        if pfx:
            lbl = r["label_num"]
            prefix_labels[pfx] = max(prefix_labels.get(pfx, 0), lbl)

    # 2. Match by (patient_id, laterality, view)
    # in cases: patient_id e.g. P_00005, left or right breast e.g. RIGHT, image view e.g. CC
    key_col = None
    for cand in ["left or right breast", "breast_side", "laterality"]:
        if cand in valid_cases.columns:
            key_col = cand
            break
    key_labels = {}
    if key_col and "patient_id" in valid_cases.columns and "image view" in valid_cases.columns:
        for (pid, side, view), group in valid_cases.groupby(["patient_id", key_col, "image view"]):
            pid_clean = str(pid).strip().upper()
            side_clean = str(side).strip().upper()[:1]  # 'R' or 'L'
            view_clean = str(view).strip().upper()
            key_labels[(pid_clean, side_clean, view_clean)] = int(group["label_num"].max())

    matched_labels = []
    unmatched_count = 0
    pattern = re.compile(r"^(?:Calc|Mass)-(?:Training|Test)_(P_\d+)_(LEFT|RIGHT)_(CC|MLO)", re.IGNORECASE)

    for _, row in meta.iterrows():
        sop = str(row.get("sop_uid", "")).strip()
        fn = str(row.get("file_name", "")).strip()
        stem = Path(fn).stem if fn else sop

        # Remove trailing _FULL_PRE or _PRE
        pfx = re.sub(r"(_FULL)?_PRE$", "", stem, flags=re.IGNORECASE)

        lbl = prefix_labels.get(pfx)
        if lbl is None and stem in prefix_labels:
            lbl = prefix_labels.get(stem)

        if lbl is None:
            m = pattern.search(stem)
            if m:
                p_id = m.group(1).upper()
                lat = m.group(2).upper()[:1]
                v = m.group(3).upper()
                lbl = key_labels.get((p_id, lat, v))

        if lbl is None:
            # Fall back to row patient_id, laterality, view
            row_pid = re.sub(r"^.*-(P_\d+).*$", r"\1", str(row.get("original_patient_id", ""))).upper()
            row_lat = str(row.get("laterality", "")).strip().upper()[:1]
            row_v = str(row.get("view", "")).strip().upper()
            lbl = key_labels.get((row_pid, row_lat, row_v))

        if lbl is not None:
            matched_labels.append(lbl)
        else:
            matched_labels.append(None)
            unmatched_count += 1

    out = meta.copy()
    out["label"] = matched_labels

    matched_total = len(out) - unmatched_count
    print(f"Matched {matched_total}/{len(out)} images to CBIS labels")
    if unmatched_count > 0:
        print(f"Dropping {unmatched_count} images without labels")
        out = out.dropna(subset=["label"]).copy()

    out["label"] = out["label"].astype(int)
    if "site" not in out.columns or (out["site"] == "").all():
        out["site"] = "cbis"
    if "category" not in out.columns:
        out["category"] = ""
    out["raw_path"] = "raw512/" + out["sop_uid"] + ".png"
    out["proc_path"] = "proc512/" + out["sop_uid"] + ".png"

    keep_cols = ["sop_uid", "patient_id"]
    if "original_patient_id" in out.columns:
        keep_cols.append("original_patient_id")
    if "image_index" in out.columns:
        keep_cols.append("image_index")
    keep_cols.extend(["laterality", "view", "label", "category", "site", "raw_path", "proc_path"])
    for extra in ["file_name", "src_path"]:
        if extra in out.columns and extra not in keep_cols:
            keep_cols.append(extra)

    labels_out = site_dir / "labels.csv"
    out[[c for c in keep_cols if c in out.columns]].to_csv(labels_out, index=False)
    print(f"Wrote {len(out)} labeled images -> {labels_out}")
    print("Label distribution:", out["label"].value_counts().to_dict(),
          "| Unique patients:", out["patient_id"].nunique() if "patient_id" in out.columns else "N/A")


if __name__ == "__main__":
    main()
