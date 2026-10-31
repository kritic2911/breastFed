#!/usr/bin/env python3
"""Join BCDR meta.csv (from ingest_png.py) with BCDR outline CSVs -> labels.csv.

Usage:
    python scripts/build_labels_bcdr.py
    python scripts/build_labels_bcdr.py --outlines data/BCDR/csv --site-dir data/bcdr

Matches images by filename (e.g. 'patient_3_study_4_img_3_4_1_LCC.tif' or file stem).
Classification mapping:
    Benign -> 0
    Malign / Malignant -> 1
If an image has multiple lesions with conflicting classifications, Malignant (1) wins.
"""
import argparse
from pathlib import Path
import pandas as pd


def parse_outlines(paths):
    dfs = []
    for p_str in paths:
        p = Path(p_str)
        if p.is_dir():
            csv_files = sorted(p.glob("*.csv"))
        elif p.is_file():
            csv_files = [p]
        else:
            csv_files = sorted(Path().glob(p_str))
        for f in csv_files:
            df = pd.read_csv(f, dtype=str).fillna("")
            df.columns = [c.strip().lower() for c in df.columns]
            dfs.append(df)
    if not dfs:
        raise SystemExit(f"No outline CSV files found in: {paths}")
    return pd.concat(dfs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Build BCDR labels.csv from meta.csv and outline annotations")
    ap.add_argument("--outlines", default="data/BCDR/csv",
                    help="directory, comma-separated list, or glob of BCDR outline CSVs (default: data/BCDR/csv)")
    ap.add_argument("--site-dir", default="data/bcdr",
                    help="directory containing meta.csv / where labels.csv will be saved (default: data/bcdr)")
    a = ap.parse_args()

    site_dir = Path(a.site_dir)
    meta_path = site_dir / "meta.csv"
    if not meta_path.exists():
        # Fall back to labels.csv if meta was saved there
        if (site_dir / "labels.csv").exists():
            meta_path = site_dir / "labels.csv"
        else:
            raise SystemExit(f"Neither meta.csv nor labels.csv found in {site_dir}. Run ingest_png.py first.")

    meta = pd.read_csv(meta_path, dtype=str).fillna("")
    if "sop_uid" not in meta.columns:
        raise SystemExit(f"Expected 'sop_uid' in {meta_path}")

    outline_paths = [s.strip() for s in a.outlines.split(",") if s.strip()]
    outlines = parse_outlines(outline_paths)

    if "image_filename" not in outlines.columns or "classification" not in outlines.columns:
        raise SystemExit(f"Expected 'image_filename' and 'classification' columns in outlines. Found: {list(outlines.columns)}")

    # Clean up outline filenames and classification
    raw_fn = outlines["image_filename"].astype(str).str.strip()
    flat_fn = raw_fn.str.replace("/", "_")
    stem = flat_fn.map(lambda x: Path(x).stem)

    cls_clean = outlines["classification"].astype(str).str.strip().str.lower()
    label_map = {"benign": 0, "malign": 1, "malignant": 1}
    labels = cls_clean.map(label_map)

    anno_df = pd.DataFrame({
        "flat_fn": flat_fn,
        "stem": stem,
        "label": labels,
    }).dropna(subset=["label"])
    anno_df["label"] = anno_df["label"].astype(int)

    # If multiple lesions exist for the same image, malignant (1) takes priority
    lab_by_stem = anno_df.groupby("stem")["label"].max().to_dict()
    lab_by_fn = anno_df.groupby("flat_fn")["label"].max().to_dict()

    # Match meta rows
    matched_labels = []
    unmatched_count = 0
    for _, row in meta.iterrows():
        sop = row.get("sop_uid", "").strip()
        fn = row.get("file_name", "").strip()
        stem_val = Path(fn).stem if fn else sop

        lbl = lab_by_stem.get(sop)
        if lbl is None and fn:
            lbl = lab_by_fn.get(fn)
        if lbl is None and stem_val:
            lbl = lab_by_stem.get(stem_val)

        if lbl is not None:
            matched_labels.append(lbl)
        else:
            matched_labels.append(None)
            unmatched_count += 1

    out = meta.copy()
    out["label"] = matched_labels

    matched_total = len(out) - unmatched_count
    print(f"Matched {matched_total}/{len(out)} images to outline annotations")
    if unmatched_count > 0:
        print(f"Dropping {unmatched_count} images without outline labels")
        out = out.dropna(subset=["label"]).copy()

    out["label"] = out["label"].astype(int)
    if "site" not in out.columns or (out["site"] == "").all():
        out["site"] = "bcdr"
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
