#!/usr/bin/env python3
"""Process a folder of DICOMs (e.g. CMMD) into 512px PNG caches + meta.csv.

    python scripts/ingest_dicom.py --dicom-dir /path/to/CMMD --site cmmd
    python scripts/ingest_dicom.py --dicom-dir /path/to/chunk --site cbis --delete   # streaming use

Outputs (under data/<site>/): raw512/, proc512/, meta.csv, failed.csv, patient_mapping.csv
Images are saved with human-readable patient-indexed IDs by default: D1-0001_1, D1-0001_2, etc.
Safe to re-run: images already cached are skipped, meta rows are de-duplicated.
Progress is saved periodically so partial runs can be safely interrupted and resumed.
Only new, unprocessed files from the folder are read and processed.
--delete removes each DICOM after it was processed successfully (for stream-and-delete).
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from common import make_proc, make_raw
from tqdm import tqdm


def load_pixels(ds):
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        if int(ds.get("SamplesPerPixel", 1)) > 1:
            arr = arr.mean(axis=-1)      # RGB -> gray
        else:
            arr = arr[0]                 # multi-frame: first frame
    slope = float(ds.get("RescaleSlope", 1) or 1)
    inter = float(ds.get("RescaleIntercept", 0) or 0)
    arr = arr * slope + inter
    if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = arr.max() - arr            # make "bright = dense tissue" consistent
    return arr


def meta_row(ds, file_path=None, mapped_pid=None, orig_pid=None, folder_pid="", uid=None, img_idx=1):
    lat = ds.get("ImageLaterality") or ds.get("Laterality") or ""
    raw_pid = orig_pid if orig_pid is not None else str(ds.get("PatientID", "")).strip()
    final_pid = mapped_pid if mapped_pid is not None else raw_pid
    final_uid = uid if uid is not None else str(ds.SOPInstanceUID)
    row = {
        "sop_uid": final_uid,
        "dicom_sop_uid": str(ds.SOPInstanceUID),
        "series_uid": str(ds.get("SeriesInstanceUID", "")),
        "patient_id": final_pid,
        "original_patient_id": raw_pid,
        "patient_folder": folder_pid,
        "image_index": img_idx,
        "laterality": str(lat).upper()[:1],
        "view": str(ds.get("ViewPosition", "")),
        "rows": str(ds.get("Rows", "")),
        "cols": str(ds.get("Columns", "")),
        "bits_stored": str(ds.get("BitsStored", "")),
        "photometric": str(ds.get("PhotometricInterpretation", "")),
        "manufacturer": str(ds.get("Manufacturer", "")),
        "model": str(ds.get("ManufacturerModelName", "")),
    }
    if file_path is not None:
        fp = Path(file_path)
        row["file_name"] = fp.name
        row["src_path"] = str(fp)
    return row


def parse_patient_map(val):
    if not val:
        return {}
    p = Path(val)
    if p.exists() and p.is_file():
        df = pd.read_csv(p, dtype=str).fillna("")
        cols = [c.strip().lower() for c in df.columns]
        orig_col, pid_col = None, None
        for c, orig in zip(cols, df.columns):
            if c in ("original_patient_id", "orig_id", "patientid", "id", "orig_patient_id", "patient_folder"):
                orig_col = orig
                break
        for c, orig in zip(cols, df.columns):
            if c in ("patient_id", "mapped_id", "target_id", "new_patient_id"):
                pid_col = orig
                break
        if orig_col and pid_col:
            return dict(zip(df[orig_col].str.strip(), df[pid_col].str.strip()))
        if len(df.columns) >= 2:
            return dict(zip(df.iloc[:, 0].str.strip(), df.iloc[:, 1].str.strip()))
    # Treat as comma-separated key=val
    m = {}
    for part in val.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            m[k.strip()] = v.strip()
    return m


def get_max_patient_idx(patient_ids, prefix="patient_id_"):
    max_idx = 0
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    fallback_pattern = re.compile(r"(\d+)$")
    for pid in patient_ids:
        pid = str(pid).strip()
        m = pattern.match(pid)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
        else:
            m2 = fallback_pattern.search(pid)
            if m2:
                max_idx = max(max_idx, int(m2.group(1)))
    return max_idx


def save_meta_and_map(meta_path, map_path, existing_df, rows, orig_to_mapped):
    if rows:
        new_df = pd.DataFrame(rows).astype(str)
        if existing_df is not None and len(existing_df):
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
        if len(combined):
            combined = combined.drop_duplicates("sop_uid", keep="last")
            combined.to_csv(meta_path, index=False)
    elif existing_df is not None and len(existing_df):
        existing_df.to_csv(meta_path, index=False)

    if orig_to_mapped:
        map_rows = []
        for orig, mapped in sorted(
            orig_to_mapped.items(),
            key=lambda x: (int(re.search(r"\d+", x[1]).group()) if re.search(r"\d+", x[1]) else 0, x[1])
        ):
            map_rows.append({"patient_id": mapped, "original_patient_id": orig})
        pd.DataFrame(map_rows).to_csv(map_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-dir", required=True)
    ap.add_argument("--site", default="cmmd")
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--delete", action="store_true",
                    help="delete each DICOM after it was processed successfully")
    ap.add_argument("--patient-prefix", nargs="?", const="patient_id_", default="patient_id_",
                    help="prefix for mapped sequential patient IDs (default: 'patient_id_')")
    ap.add_argument("--keep-original-patient-id", action="store_true",
                    help="do not map patient IDs to patient_id_1, patient_id_2, etc.")
    ap.add_argument("--patient-map", help="CSV file or mapping string (orig_id/folder -> patient_id) for manual mapping")
    ap.add_argument("--uid-format", choices=["patient_seq", "stem", "sop"], default="patient_seq",
                    help="image ID format: patient_seq (e.g. D1-0001_1, D1-0001_2), stem (file stem), or sop (DICOM SOP UID)")
    a = ap.parse_args()

    out = Path(a.out_root) / a.site
    raw_dir, proc_dir = out / "raw512", out / "proc512"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.csv"
    map_path = out / "patient_mapping.csv"

    # 1. Load existing metadata and mapping if available
    existing_df = None
    _processed_sop = set()
    processed_src_paths = set()
    processed_src_resolved = set()
    orig_to_mapped = {}
    patient_image_counts = Counter()

    if meta_path.exists():
        existing_df = pd.read_csv(meta_path, dtype=str).fillna("")
        if "sop_uid" in existing_df.columns:
            _processed_sop = set(existing_df["sop_uid"].dropna())
            for sop in existing_df["sop_uid"].dropna():
                m = re.match(r"^(.*)_(\d+)$", str(sop).strip())
                if m:
                    pid, seq = m.group(1), int(m.group(2))
                    patient_image_counts[pid] = max(patient_image_counts[pid], seq)
        if "src_path" in existing_df.columns:
            processed_src_paths = {p.strip() for p in existing_df["src_path"].dropna() if p.strip()}
            processed_src_resolved = {str(Path(p).resolve()) for p in processed_src_paths}
        if "original_patient_id" in existing_df.columns and "patient_id" in existing_df.columns:
            for _, r in existing_df.iterrows():
                op, mp = r["original_patient_id"].strip(), r["patient_id"].strip()
                if op and mp:
                    orig_to_mapped[op] = mp

    if map_path.exists():
        map_df = pd.read_csv(map_path, dtype=str).fillna("")
        if "original_patient_id" in map_df.columns and "patient_id" in map_df.columns:
            for _, r in map_df.iterrows():
                op, mp = r["original_patient_id"].strip(), r["patient_id"].strip()
                if op and mp:
                    orig_to_mapped[op] = mp

    if a.patient_map:
        manual_map = parse_patient_map(a.patient_map)
        orig_to_mapped.update(manual_map)

    patient_prefix = a.patient_prefix or "patient_id_"
    patient_counter = get_max_patient_idx(orig_to_mapped.values(), prefix=patient_prefix)

    # 2. Find files in dicom_dir
    src = Path(a.dicom_dir)
    files = sorted(src.rglob("*.dcm"))
    if not files:  # some downloads have no extension
        files = sorted(p for p in src.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"No files found under {src}")
    print(f"{len(files)} files found under {src}")

    # 3. Check for files already processed
    new_files = []
    skipped_processed = 0
    for f in files:
        f_str = str(f)
        f_res = str(f.resolve())
        if f_str in processed_src_paths or f_res in processed_src_resolved:
            skipped_processed += 1
            continue
        new_files.append(f)

    if skipped_processed > 0:
        print(f"Skipping {skipped_processed} already processed files. {len(new_files)} new files to process.")

    if not new_files:
        print(f"All {len(files)} files have already been processed.")
        if existing_df is not None and len(existing_df):
            print(f"meta.csv has {len(existing_df)} images, {existing_df['patient_id'].nunique()} patients")
        return

    # 4. Process new files
    rows, failed = [], []
    try:
        for f in tqdm(new_files, desc=a.site):
            try:
                # Extract patient folder name (e.g. D1-0001 from data/cmmd/images/D1-0001/...)
                rel_parts = f.relative_to(src).parts
                folder_pid = rel_parts[0] if len(rel_parts) > 1 else ""

                ds = pydicom.dcmread(f)
                sop_uid = str(ds.SOPInstanceUID)
                dicom_pid = str(ds.get("PatientID", "")).strip()
                orig_pid = folder_pid or dicom_pid or "unknown"

                patient_image_counts[orig_pid] += 1
                img_idx = patient_image_counts[orig_pid]

                if a.uid_format == "patient_seq":
                    uid = f"{orig_pid}_{img_idx}"
                elif a.uid_format == "stem":
                    uid = f.stem
                else:
                    uid = sop_uid

                if a.keep_original_patient_id:
                    mapped_pid = orig_pid
                else:
                    if orig_pid not in orig_to_mapped:
                        patient_counter += 1
                        orig_to_mapped[orig_pid] = f"{patient_prefix}{patient_counter}"
                    mapped_pid = orig_to_mapped[orig_pid]

                rp, pp = raw_dir / f"{uid}.png", proc_dir / f"{uid}.png"
                if not (rp.exists() and pp.exists()):
                    arr = load_pixels(ds)
                    cv2.imwrite(str(rp), make_raw(arr))
                    cv2.imwrite(str(pp), make_proc(arr))
                rows.append(meta_row(ds, file_path=f, mapped_pid=mapped_pid, orig_pid=orig_pid,
                                     folder_pid=folder_pid, uid=uid, img_idx=img_idx))
                if a.delete:
                    f.unlink()

                # Periodically flush metadata so progress is never lost
                if len(rows) % 50 == 0:
                    save_meta_and_map(meta_path, map_path, existing_df, rows, orig_to_mapped)

            except Exception as e:  # keep going; failures are logged
                failed.append({"file": str(f), "error": repr(e)})
    finally:
        # Save on normal completion, error, or KeyboardInterrupt (Ctrl+C)
        save_meta_and_map(meta_path, map_path, existing_df, rows, orig_to_mapped)

    if failed:
        fp = out / "failed.csv"
        pd.DataFrame(failed).to_csv(fp, mode="a", header=not fp.exists(), index=False)
        print(f"WARNING: {len(failed)} files failed, see {fp}. First error: {failed[0]['error']}")
        print("If the error mentions decoding/transfer syntax, run: pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg")

    if a.delete:  # remove now-empty folders
        for d in sorted(src.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    total_images = (len(existing_df) if existing_df is not None else 0) + len(rows)
    print(f"done: {len(rows)} new processed, {len(failed)} failed -> {out}")
    print(f"meta.csv now has {total_images} images, {len(orig_to_mapped)} patients")
    print(f"patient mapping saved to {map_path}")


if __name__ == "__main__":
    main()
