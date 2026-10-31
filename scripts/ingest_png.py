#!/usr/bin/env python3
"""Ingest a folder of PNG/JPG/TIF images (e.g. INbreast, VinDr, BCDR) into the same
512px cache format as the DICOM sites, and write labels.csv or meta.csv.

Mode A - Folder-sorted images with explicit folder -> label map:
    python scripts/ingest_png.py --root /path/to/inbreast_png --site inbreast \
        --map "birads1=drop,birads2=0,birads3=0,birads4=1,birads5=1" \
        --patient-regex "^(\\d+)_"

Mode B - Metadata-only ingestion without --map (generates meta.csv; e.g. for BCDR):
    python scripts/ingest_png.py --root data/BCDR/BCDR-Images-Original/ --site bcdr
    python scripts/build_labels_bcdr.py --outlines data/BCDR/csv --site-dir data/bcdr

Safe to re-run: checks for files already processed and only processes new files.
Progress is saved periodically so partial runs can be safely interrupted and resumed.
Images are saved with patient-indexed IDs or file stems (BCDR .tif automatically uses file stem).
Patient IDs are stored as patient_id_1, patient_id_2, ... along with original IDs, and an
explicit patient_mapping.csv is created/updated for easy manual mapping and auditing.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from common import make_proc, make_raw
from tqdm import tqdm

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_map(s):
    m = {}
    for part in s.split(","):
        k, v = part.split("=")
        m[k.strip().lower()] = None if v.strip().lower() == "drop" else int(v)
    return m


def category(root, p):
    parts = p.relative_to(root).parts
    return parts[0] if len(parts) > 1 else ""


def read_gray(p):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("unreadable image")
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    return img.astype(np.float32)


def parse_patient_map(val):
    if not val:
        return {}
    p = Path(val)
    if p.exists() and p.is_file():
        df = pd.read_csv(p, dtype=str).fillna("")
        cols = [c.strip().lower() for c in df.columns]
        orig_col, pid_col = None, None
        for c, orig in zip(cols, df.columns):
            if c in ("original_patient_id", "orig_id", "patientid", "id", "orig_patient_id", "file_name", "image"):
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


def save_labels_and_map(labels_path, map_path, existing_df, rows, orig_to_mapped):
    if rows:
        new_df = pd.DataFrame(rows).astype(str)
        if existing_df is not None and len(existing_df):
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
        if len(combined):
            combined = combined.drop_duplicates("sop_uid", keep="last")
            combined.to_csv(labels_path, index=False)
    elif existing_df is not None and len(existing_df):
        existing_df.to_csv(labels_path, index=False)

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
    ap.add_argument("--root", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--list", action="store_true", help="only list folders, counts and sample names")
    ap.add_argument("--map", help='e.g. "benign=0,malignant=1,normal=drop"')
    ap.add_argument("--patient-regex", help="group 1 = patient/study id, matched on filename stem")
    ap.add_argument("--patient-prefix", nargs="?", const="patient_id_", default="patient_id_",
                    help="prefix for mapped sequential patient IDs (default: 'patient_id_')")
    ap.add_argument("--patient-map", help="CSV file or mapping string (orig_id/filename -> patient_id) for manual mapping")
    ap.add_argument("--keep-original-patient-id", action="store_true",
                    help="keep raw regex/filename as patient_id instead of patient_id_1, patient_id_2, ...")
    ap.add_argument("--uid-format", choices=["patient_seq", "stem"], default="patient_seq",
                    help="image ID format: patient_seq (e.g. patient_1, patient_2), or stem (file stem; automatic for BCDR .tif)")
    a = ap.parse_args()

    root = Path(a.root)
    files = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in IMG_EXT and not any(part in ("raw512", "proc512") for part in p.parts)
    )
    if not files:
        raise SystemExit(f"No images found under {root}")

    if a.list:
        cats = Counter(category(root, p) for p in files)
        for c, n in sorted(cats.items()):
            ex = [p.stem for p in files if category(root, p) == c][:3]
            print(f"{c or '(top level)'!r:30} {n:6d} images   e.g. {ex}")
        print("\nUse these folder names in --map. Check the sample names to write --patient-regex.")
        return

    cmap = parse_map(a.map) if a.map else None
    rx = re.compile(a.patient_regex) if a.patient_regex else None
    if rx is None and "bcdr" in a.site.lower():
        rx = re.compile(r"^patient_(\d+)")
    elif rx is None and "cbis" in a.site.lower():
        rx = re.compile(r"_(P_\d+)_")
    manual_map = parse_patient_map(a.patient_map) if a.patient_map else {}
    inbreast_info = {}
    if ("inbreast" in a.site.lower() or "inbreast" in str(root).lower()) and not a.patient_map:
        dcm_dir = None
        for cand in [root / "documentation" / "AllDICOMs", Path("data/INBreast/documentation/AllDICOMs"), Path("data/inbreast/documentation/AllDICOMs")]:
            if cand.exists():
                dcm_dir = cand
                break
        if dcm_dir:
            for dcm_file in dcm_dir.glob("*.dcm"):
                parts = dcm_file.stem.split("_")
                if len(parts) >= 5:
                    img_id, d_pid, d_lat, d_view = parts[0], parts[1], parts[3], parts[4]
                    if img_id not in manual_map:
                        manual_map[img_id] = f"{a.site}-{d_pid}"
                    inbreast_info[img_id] = (d_lat, "MLO" if d_view == "ML" else d_view)

    if rx is None and not manual_map:
        print("NOTE: no --patient-regex or --patient-map provided; each unique image will be assigned its own patient_id.")

    out = Path(a.out_root) / a.site
    raw_dir, proc_dir = out / "raw512", out / "proc512"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out / "labels.csv"
    meta_path = out / "meta.csv"
    csv_path = labels_path if a.map else meta_path
    map_path = out / "patient_mapping.csv"

    # 1. Load existing labels/meta and patient mapping if present
    existing_df = None
    processed_sop = set()
    processed_src_paths = set()
    processed_src_resolved = set()
    orig_to_mapped = {}
    patient_image_counts = Counter()

    for check_path in [csv_path, labels_path, meta_path]:
        if check_path.exists():
            df_curr = pd.read_csv(check_path, dtype=str).fillna("")
            if existing_df is None and check_path == csv_path:
                existing_df = df_curr
            if "sop_uid" in df_curr.columns:
                processed_sop.update(df_curr["sop_uid"].dropna())
                for sop in df_curr["sop_uid"].dropna():
                    m = re.match(r"^(.*)_(\d+)$", str(sop).strip())
                    if m:
                        pid, seq = m.group(1), int(m.group(2))
                        patient_image_counts[pid] = max(patient_image_counts[pid], seq)
            if "src_path" in df_curr.columns:
                srcs = {p.strip() for p in df_curr["src_path"].dropna() if p.strip()}
                processed_src_paths.update(srcs)
                processed_src_resolved.update({str(Path(p).resolve()) for p in srcs})
            if "original_patient_id" in df_curr.columns and "patient_id" in df_curr.columns:
                for _, r in df_curr.iterrows():
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

    patient_prefix = a.patient_prefix or "patient_id_"
    patient_counter = get_max_patient_idx(orig_to_mapped.values(), prefix=patient_prefix)

    # 2. Check for files already processed
    new_files = []
    skipped_processed = 0
    for p in files:
        p_str = str(p)
        p_res = str(p.resolve())
        if p_str in processed_src_paths or p_res in processed_src_resolved:
            skipped_processed += 1
            continue
        new_files.append(p)

    if skipped_processed > 0:
        print(f"Skipping {skipped_processed} already processed images. {len(new_files)} new images to process.")

    if not new_files:
        print(f"All {len(files)} images have already been processed.")
        if existing_df is not None and len(existing_df):
            print("label counts:", existing_df["label"].value_counts().to_dict(),
                  "| patients:", existing_df["patient_id"].nunique())
        return

    # 3. Process new images
    rows, all_meta_rows, failed, skipped, no_match = [], [], [], Counter(), 0
    seen_uids = set(processed_sop)

    try:
        for p in tqdm(new_files, desc=a.site):
            cat = category(root, p)
            dropped = False
            if cmap is not None:
                cat_key = cat.lower()
                if cat_key not in cmap:
                    if not cat_key and "(top level)" in cmap:
                        cat_key = "(top level)"
                    elif not cat_key and "top level" in cmap:
                        cat_key = "top level"
                    else:
                        skipped[cat or "(top level)"] += 1
                        continue
                label = cmap[cat_key]
                if label is None:
                    skipped[f"{cat or '(top level)'} (dropped)"] += 1
                    dropped = True
                    label = ""
            else:
                label = ""

            # Determine patient identifier
            if p.stem in manual_map:
                orig_pid = manual_map[p.stem]
            elif p.name in manual_map:
                orig_pid = manual_map[p.name]
            elif rx is not None:
                m = rx.search(p.stem)
                if m:
                    orig_pid = f"{a.site}-{m.group(1)}"
                else:
                    orig_pid = f"{a.site}-{p.stem}"
                    no_match += 1
            else:
                orig_pid = f"{a.site}-{p.stem}"

            # Map to sequential patient ID (patient_id_1, patient_id_2, ...)
            if a.keep_original_patient_id:
                pid = orig_pid
            else:
                if orig_pid not in orig_to_mapped:
                    patient_counter += 1
                    orig_to_mapped[orig_pid] = f"{patient_prefix}{patient_counter}"
                pid = orig_to_mapped[orig_pid]

            patient_image_counts[orig_pid] += 1
            img_idx = patient_image_counts[orig_pid]

            site_lower = a.site.lower()
            use_stem_uid = (
                a.uid_format == "stem"
                or (p.suffix.lower() in {".tif", ".tiff"} and "bcdr" in site_lower)
                or "cbis" in site_lower
                or "inbreast" in site_lower
                or "bcdr" in site_lower
            )

            if use_stem_uid:
                uid = p.stem
            else:
                uid = f"{orig_pid}_{img_idx}"

            if uid in seen_uids:
                uid = f"{uid}_{len(seen_uids)}"
            seen_uids.add(uid)

            # Detect laterality and view from filename or metadata if present
            lat, view = "", ""
            if p.stem in inbreast_info:
                lat, view = inbreast_info[p.stem]
            else:
                stem_upper = p.stem.upper()
                if "_L_" in stem_upper or stem_upper.startswith("L_") or "_LEFT" in stem_upper or "_LCC" in stem_upper or "_LO" in stem_upper:
                    lat = "L"
                elif "_R_" in stem_upper or stem_upper.startswith("R_") or "_RIGHT" in stem_upper or "_RCC" in stem_upper or "_RO" in stem_upper:
                    lat = "R"
                if "CC" in stem_upper:
                    view = "CC"
                elif "MLO" in stem_upper or "ML" in stem_upper or stem_upper.endswith("_LO") or stem_upper.endswith("_RO") or "_LO_" in stem_upper or "_RO_" in stem_upper:
                    view = "MLO"

            try:
                rp, pp = raw_dir / f"{uid}.png", proc_dir / f"{uid}.png"
                if not (rp.exists() and pp.exists()):
                    arr = read_gray(p)
                    cv2.imwrite(str(rp), make_raw(arr))
                    cv2.imwrite(str(pp), make_proc(arr))
                row_data = {
                    "sop_uid": uid,
                    "patient_id": pid,
                    "original_patient_id": orig_pid,
                    "image_index": img_idx,
                    "laterality": lat,
                    "view": view,
                    "label": label,
                    "category": cat,
                    "site": a.site,
                    "raw_path": f"raw512/{uid}.png",
                    "proc_path": f"proc512/{uid}.png",
                    "file_name": p.name,
                    "src_path": str(p),
                }
                all_meta_rows.append(row_data)
                if not dropped:
                    rows.append(row_data)

                if len(all_meta_rows) % 50 == 0:
                    save_labels_and_map(csv_path, map_path, existing_df, rows, orig_to_mapped)
                    if a.map:
                        save_labels_and_map(meta_path, map_path, None, all_meta_rows, orig_to_mapped)

            except Exception as e:
                failed.append({"file": str(p), "error": repr(e)})
    finally:
        save_labels_and_map(csv_path, map_path, existing_df, rows, orig_to_mapped)
        if a.map and all_meta_rows:
            save_labels_and_map(meta_path, map_path, None, all_meta_rows, orig_to_mapped)

    if failed:
        pd.DataFrame(failed).to_csv(out / "failed.csv", index=False)

    total_images = (len(existing_df) if existing_df is not None else 0) + len(rows)
    print(f"\nwrote {total_images} total images -> {csv_path} ({len(rows)} new, {len(failed)} failed)")
    if a.map and all_meta_rows:
        print(f"wrote {len(all_meta_rows)} total images -> {meta_path}")
    if skipped:
        print("skipped:", dict(skipped))
    if rx is not None and no_match:
        print(f"WARNING: {no_match} filenames did not match --patient-regex")
    print(f"patients: {len(orig_to_mapped)}")
    print(f"patient mapping saved to {map_path}")


if __name__ == "__main__":
    main()
