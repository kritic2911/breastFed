#!/usr/bin/env python3
"""Generate a preview of 10 randomly selected images from each site for human verification.

Adheres strictly to AGENTS.md Phase P1 & QC rules:
  - Selects 10 stratified random images per site (vindr, cmmd, inbreast, cdd).
  - Displays cached 512x512 images with standardized orientation (chest wall left, nipple right).
  - Overlays lesion annotations (green bounding boxes) where available.
  - Displays format, bit depth, class label, patient hash, view, and laterality.
  - Saves grids to paper/figures/qc_grids/ and copies to the artifact directory.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

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

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
Path("/tmp/matplotlib_cache").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path("/home/kc/.gemini/antigravity/brain/c89e4b79-4363-4661-b21d-d54255b7a2b7")


def generate_site_10_previews():
    manifest_csv = Path("data/manifest.csv")
    annotations_csv = Path("data/annotations.csv")
    data_dir = Path("data")
    out_dir = Path("paper/figures/qc_grids")
    out_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not manifest_csv.exists():
        logger.error("data/manifest.csv does not exist!")
        sys.exit(1)

    mdf = pd.read_csv(manifest_csv)
    adf = pd.read_csv(annotations_csv) if annotations_csv.exists() else pd.DataFrame()

    label_names = {0: "0: Normal", 1: "1: Benign", 2: "2: Suspicious/Mal"}
    site_names = {
        "vindr": "VinDr-Mammo (mini)",
        "cmmd": "CMMD",
        "inbreast": "INbreast",
        "cdd": "CDD-CESM (Low Energy DM)",
    }

    all_preview_records = []

    for site in ["vindr", "cmmd", "inbreast", "cdd"]:
        site_df = mdf[mdf["site"] == site].copy()
        if site_df.empty:
            logger.warning("No images found for site %s", site)
            continue

        # Stratified sampling of 10 images across available classes
        classes = sorted(site_df["label"].unique())
        n_per_class = max(1, 10 // len(classes))
        samples = []
        for c in classes:
            c_df = site_df[site_df["label"] == c]
            n_sel = min(n_per_class, len(c_df))
            samples.append(c_df.sample(n=n_sel, random_state=42))

        sample_df = pd.concat(samples)
        if len(sample_df) < 10:
            rem = site_df[~site_df["image_id"].isin(sample_df["image_id"])]
            n_need = 10 - len(sample_df)
            if n_need > 0 and not rem.empty:
                sample_df = pd.concat([sample_df, rem.sample(n=min(n_need, len(rem)), random_state=42)])

        sample_df = sample_df.head(10).reset_index(drop=True)

        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        fig.suptitle(
            f"10-Image Verification Preview: {site_names.get(site, site.upper())} "
            f"(512x512, Standard Laterality: Chest Wall Left)",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        for idx, ax in enumerate(axes.flat):
            if idx >= len(sample_df):
                ax.axis("off")
                continue

            row = sample_df.iloc[idx]
            img_id = row["image_id"]
            p_hash = row["patient_hash"]
            cached_rel = row.get("cached_path")
            cached_path = data_dir / cached_rel if cached_rel and not pd.isna(cached_rel) else None

            if cached_path and cached_path.exists():
                arr = np.load(cached_path)
            else:
                arr = np.zeros((512, 512), dtype=np.float32)

            ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)

            # Draw annotation boxes if available
            n_boxes = 0
            if not adf.empty:
                site_ann = adf[(adf["site"] == site) & (adf["image_id"] == img_id)]
                for _, b in site_ann.iterrows():
                    x1, y1 = b["cached_xmin"], b["cached_ymin"]
                    x2, y2 = b["cached_xmax"], b["cached_ymax"]
                    if not (pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2)):
                        rect = plt.Rectangle(
                            (x1, y1),
                            max(2.0, x2 - x1),
                            max(2.0, y2 - y1),
                            fill=False,
                            edgecolor="#00FF00",
                            linewidth=2.0,
                        )
                        ax.add_patch(rect)
                        n_boxes += 1

            lbl_txt = label_names.get(int(row["label"]), str(row["label"]))
            raw_desc = str(row.get("birads_raw", ""))
            if raw_desc == "nan" or not raw_desc:
                raw_desc = f"Class {row['label']}"
            fmt_str = f"{str(row.get('source_format', 'dcm')).upper()} ({row.get('bit_depth', 16)}b)"
            tissue_pct = f"{row.get('tissue_pct', 0.0):.1f}%"

            ann_tag = f" | {n_boxes} lesion box" if n_boxes > 0 else ""
            title = (
                f"#{idx+1}: {p_hash[:8]}.. | {row['laterality']}-{row['view']}\n"
                f"{lbl_txt} [{row['split']}]{ann_tag}\n"
                f"{fmt_str} | Tissue: {tissue_pct}"
            )
            ax.set_title(title, fontsize=8.5)
            ax.axis("off")

            all_preview_records.append({
                "site": site,
                "index": idx + 1,
                "patient_hash_prefix": p_hash[:8],
                "laterality_view": f"{row['laterality']}-{row['view']}",
                "label": int(row["label"]),
                "label_name": lbl_txt,
                "raw_finding": raw_desc,
                "split": row["split"],
                "format": str(row.get("source_format", "dcm")).upper(),
                "bit_depth": int(row.get("bit_depth", 16)),
                "tissue_pct": float(row.get("tissue_pct", 0.0)),
                "lesion_boxes": n_boxes,
            })

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_png = out_dir / f"preview_{site}_10.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        logger.info("Saved preview grid: %s", out_png)

        # Copy to artifact directory
        artifact_png = ARTIFACT_DIR / f"preview_{site}_10.png"
        shutil.copyfile(out_png, artifact_png)
        logger.info("Copied to artifact directory: %s", artifact_png)

    # Save summary table
    preview_df = pd.DataFrame(all_preview_records)
    preview_csv = out_dir / "preview_samples_meta.csv"
    preview_df.to_csv(preview_csv, index=False)
    logger.info("Saved preview metadata to %s", preview_csv)


if __name__ == "__main__":
    generate_site_10_previews()

