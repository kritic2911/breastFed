#!/usr/bin/env python3
"""Build unified manifest, preprocess mammograms to 512x512, and cache arrays.

Adheres strictly to AGENTS.md Phase P1 & Rules 6, 9:
  - Generates unified manifest.csv across all 4 sites.
  - Preprocesses each image: VOI LUT, MONOCHROME1 inversion, breast crop,
    standard laterality flip (chest wall left), tissue percentile normalization,
    and aspect-preserving resize + pad to 512x512.
  - Caches float32 arrays in data/cache512/<patient_hash>_<site>_<lat>_<view>.npy.
  - Computes and records SHA-256 checksum of the cache.
  - Builds canonical lesion annotations table (data/annotations.csv) in 512x512 space.
"""

import hashlib
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

# Auto-delegate to local .venv if run with system python
_repo_dir = Path(__file__).resolve().parent.parent
_venv_dir = _repo_dir / ".venv"
_venv_py = _venv_dir / "bin" / "python3"
if _venv_py.exists() and sys.prefix != str(_venv_dir):
    try:
        import pandas  # noqa: F401
    except ImportError:
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

if str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

import numpy as np
import pandas as pd

from src.data.annotations import build_annotations_table
from src.data.manifest import build_manifest
from src.data.preprocess import preprocess_mammogram, read_image_array

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_single_image(row_dict: Dict[str, Any], cache_dir: Path) -> Dict[str, Any]:
    """Worker function to process and cache one image."""
    img_id = row_dict["image_id"]
    p_hash = row_dict["patient_hash"]
    site = row_dict["site"]
    lat = row_dict["laterality"]
    view = row_dict["view"]
    raw_path = row_dict["path"]

    # Target cache filename: <p_hash>_<site>_<lat>_<view>_<img_id>.npy
    cached_filename = f"{p_hash}_{site}_{lat}_{view}_{img_id}.npy"
    cached_path = cache_dir / cached_filename

    try:
        arr, info = read_image_array(raw_path)
        c_img, meta = preprocess_mammogram(arr, target_size=512)

        # Save float32 array atomically
        tmp_path = cached_path.with_suffix(".tmp.npy")
        np.save(tmp_path, c_img.astype(np.float32))
        tmp_path.replace(cached_path)

        return {
            "path": raw_path,
            "image_id": img_id,
            "site": site,
            "source_format": info["source_format"],
            "bit_depth": info["bit_depth"],
            "crop_box": meta["crop_box_str"],
            "flip": meta["flip"],
            "scale": meta["scale"],
            "original_shape": meta["original_shape_str"],
            "tissue_pct": meta["tissue_pct"],
            "cached_path": str(cached_path.relative_to(cache_dir.parent)),
            "status": "success",
        }
    except Exception as e:
        logger.error("Error processing %s (%s): %s", img_id, raw_path, e)
        return {
            "path": raw_path,
            "image_id": img_id,
            "site": site,
            "source_format": None,
            "bit_depth": None,
            "crop_box": None,
            "flip": None,
            "scale": None,
            "original_shape": None,
            "tissue_pct": None,
            "cached_path": None,
            "status": f"failed: {e}",
        }


def compute_directory_sha256(directory: Path) -> str:
    """Compute deterministic SHA-256 checksum of all files in a directory."""
    files = sorted(directory.glob("*.npy"))
    hasher = hashlib.sha256()
    for f in files:
        hasher.update(f.name.encode("utf-8"))
        hasher.update(str(f.stat().st_size).encode("utf-8"))
    return hasher.hexdigest()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build manifest and cache 512x512 images.")
    parser.add_argument("--force", action="store_true", help="Force reprocessing and re-caching of all images.")
    args = parser.parse_args()

    start_time = time.time()
    data_dir = Path("data")
    cache_dir = data_dir / "cache512"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        logger.info("Force flag specified: clearing existing cached files ...")
        for f in cache_dir.glob("*.npy"):
            try:
                f.unlink()
            except Exception:
                pass

    manifest_csv = data_dir / "manifest.csv"
    annotations_csv = data_dir / "annotations.csv"

    logger.info("=== Phase P1: Building unified manifest ===")
    manifest_df = build_manifest(data_dir=data_dir)
    total_images = len(manifest_df)
    logger.info("Total images to process: %d", total_images)

    # Determine which images need processing
    rows_to_process = []
    cached_existing = 0
    for row in manifest_df.to_dict("records"):
        img_id = row["image_id"]
        p_hash = row["patient_hash"]
        site = row["site"]
        lat = row["laterality"]
        view = row["view"]
        cached_file = cache_dir / f"{p_hash}_{site}_{lat}_{view}_{img_id}.npy"
        if cached_file.exists() and cached_file.stat().st_size > 0:
            cached_existing += 1
        else:
            rows_to_process.append(row)

    logger.info("Existing cached files: %d, files to process: %d", cached_existing, len(rows_to_process))

    # Parallel processing
    results = {}
    num_workers = max(1, os.cpu_count() or 4)
    logger.info("Launching parallel worker pool with %d workers ...", num_workers)

    if rows_to_process:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_id = {
                executor.submit(process_single_image, r, cache_dir): r["path"]
                for r in rows_to_process
            }

            done_count = 0
            for future in as_completed(future_to_id):
                res = future.result()
                key = res["path"]
                results[key] = res
                done_count += 1
                if done_count % 500 == 0 or done_count == len(rows_to_process):
                    logger.info("Processed %d / %d images (%.1f%%)", done_count, len(rows_to_process), done_count / len(rows_to_process) * 100)

    # Update manifest DataFrame with results
    updated_rows = []
    for r in manifest_df.to_dict("records"):
        key = r["path"]
        if key in results and results[key]["status"] == "success":
            res = results[key]
            r["source_format"] = res.get("source_format", r.get("source_format"))
            r["bit_depth"] = res.get("bit_depth", r.get("bit_depth"))
            r["crop_box"] = res["crop_box"]
            r["flip"] = res["flip"]
            r["scale"] = res["scale"]
            r["original_shape"] = res["original_shape"]
            r["tissue_pct"] = res["tissue_pct"]
            r["cached_path"] = res["cached_path"]
        else:
            # Check existing file
            img_id = r["image_id"]
            p_hash = r["patient_hash"]
            site = r["site"]
            lat = r["laterality"]
            view = r["view"]
            cached_file = cache_dir / f"{p_hash}_{site}_{lat}_{view}_{img_id}.npy"
            if cached_file.exists():
                r["cached_path"] = str(cached_file.relative_to(data_dir))
        updated_rows.append(r)

    manifest_df = pd.DataFrame(updated_rows)

    # Filter out any failed image if any
    valid_manifest = manifest_df[manifest_df["cached_path"].notna()].copy()
    logger.info("Final valid cached images in manifest: %d / %d", len(valid_manifest), total_images)

    # Save manifest.csv
    valid_manifest.to_csv(manifest_csv, index=False)
    logger.info("Saved final manifest.csv with %d rows to %s", len(valid_manifest), manifest_csv)

    # Compute cache checksum
    cache_checksum = compute_directory_sha256(cache_dir)
    logger.info("Cache SHA-256 Checksum: %s", cache_checksum)

    # Build annotations.csv
    logger.info("=== Phase P1: Building lesion annotations table ===")
    ann_df = build_annotations_table(valid_manifest, data_dir=data_dir, output_csv=annotations_csv)
    logger.info("Built annotations table with %d annotations", len(ann_df))

    elapsed = time.time() - start_time
    logger.info("Phase P1 preprocessing & caching completed in %.1f seconds", elapsed)

    print("\n" + "=" * 60)
    print(f"CACHE COMPLETE: {len(valid_manifest)} images cached at 512x512")
    print(f"CACHE CHECKSUM: {cache_checksum}")
    print(f"MANIFEST: {manifest_csv}")
    print(f"ANNOTATIONS: {annotations_csv}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
