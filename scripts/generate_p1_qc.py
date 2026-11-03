#!/usr/bin/env python3
"""QC visualization grids and tissue intensity histograms for Phase P1.

Adheres strictly to AGENTS.md Phase P1 & Checkpoint P1:
  - Generates 20-sample QC grids per site in paper/figures/qc_grids/<site>_qc_grid.png
    (human verifies MONOCHROME inversion, standard chest wall orientation, and markers).
  - Plots per-site tissue intensity histograms in paper/figures/qc_grids/intensity_histograms.png.
  - Verifies that histograms are free of identical-position spike artifacts.
"""

import logging
import os
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


def generate_site_qc_grids(manifest_df: pd.DataFrame, ann_df: pd.DataFrame, out_dir: Path):
    """Generate 4x5 QC grids of 20 sample images per site with metadata and annotations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data")

    label_names = {0: "Normal", 1: "Benign", 2: "Suspicious/Malignant"}

    for site, group in manifest_df.groupby("site"):
        logger.info("Generating QC grid for site: %s ...", site)
        # Stratified sample across available classes
        samples = []
        classes = group["label"].unique()
        per_class = max(1, 20 // len(classes))
        for c in classes:
            c_group = group[group["label"] == c]
            n_select = min(per_class, len(c_group))
            samples.append(c_group.sample(n=n_select, random_state=42))

        sample_df = pd.concat(samples)
        if len(sample_df) < 20:
            remaining = group[~group["image_id"].isin(sample_df["image_id"])]
            n_more = min(20 - len(sample_df), len(remaining))
            if n_more > 0:
                sample_df = pd.concat([sample_df, remaining.sample(n=n_more, random_state=42)])
        sample_df = sample_df.head(20).reset_index(drop=True)

        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        fig.suptitle(f"QC Grid: {site.upper()} (20 Samples, Chest Wall Left, 512x512)", fontsize=16, y=0.98)

        for idx, ax in enumerate(axes.flat):
            if idx >= len(sample_df):
                ax.axis("off")
                continue

            row = sample_df.iloc[idx]
            img_id = row["image_id"]
            cached_path = data_dir / row["cached_path"]

            if cached_path.exists():
                arr = np.load(cached_path)
            else:
                arr = np.zeros((512, 512), dtype=np.float32)

            ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)

            # Check for annotations to overlay
            if not ann_df.empty:
                boxes = ann_df[(ann_df["site"] == site) & (ann_df["image_id"] == img_id)]
                for _, b in boxes.iterrows():
                    x1, y1 = b["cached_xmin"], b["cached_ymin"]
                    x2, y2 = b["cached_xmax"], b["cached_ymax"]
                    if not (pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2)):
                        rect = plt.Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            fill=False,
                            edgecolor="#00FF00",
                            linewidth=1.5,
                        )
                        ax.add_patch(rect)

            lbl_str = label_names.get(int(row["label"]), str(row["label"]))
            split_str = str(row["split"])
            title = f"{row['patient_hash'][:8]}.. | {row['view']} {row['laterality']}\n{lbl_str} ({split_str})"
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_file = out_dir / f"{site}_qc_grid.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out_file)


def generate_intensity_histograms(manifest_df: pd.DataFrame, out_file: Path):
    """Plot per-site tissue intensity histograms and assert no identical-position spikes."""
    data_dir = Path("data")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Phase P1: Per-Site Tissue-Intensity Histograms (512x512 Cached Images)", fontsize=15, y=0.98)

    palette = {"vindr": "#1f77b4", "cmmd": "#ff7f0e", "inbreast": "#2ca02c", "cdd": "#d62728"}
    bins = np.linspace(0.01, 1.0, 100)

    site_peaks = {}
    for (site, group), ax in zip(manifest_df.groupby("site"), axes.flat):
        logger.info("Computing intensity histogram for site: %s ...", site)
        # Sample up to 100 images per site for histogram computation
        sample = group.sample(n=min(100, len(group)), random_state=42)
        tissue_pixels = []

        for _, row in sample.iterrows():
            p = data_dir / row["cached_path"]
            if p.exists():
                arr = np.load(p)
                # Sample non-zero tissue pixels
                t_pix = arr[arr > 0.02]
                if len(t_pix) > 0:
                    tissue_pixels.append(np.random.choice(t_pix, size=min(1000, len(t_pix)), replace=False))

        if tissue_pixels:
            all_pix = np.concatenate(tissue_pixels)
            counts, bin_edges, _ = ax.hist(
                all_pix, bins=bins, density=True, alpha=0.75, color=palette.get(site, "blue"), edgecolor="none"
            )
            # Find major peak position (excluding near 0)
            peak_bin_idx = int(np.argmax(counts[5:])) + 5
            peak_intensity = 0.5 * (bin_edges[peak_bin_idx] + bin_edges[peak_bin_idx + 1])
            site_peaks[site] = peak_intensity

            ax.set_title(f"Site: {site.upper()} (n={len(sample)} images)\nPeak intensity: {peak_intensity:.3f}", fontsize=11)
            ax.set_xlabel("Normalized Pixel Intensity [0, 1]", fontsize=9)
            ax.set_ylabel("Probability Density", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info("Saved intensity histograms to %s", out_file)

    # Verification: assert peak positions are diverse (no identical-position spike artifact)
    peaks = list(site_peaks.values())
    logger.info("Per-site peak tissue intensities: %s", site_peaks)
    assert len(set(np.round(peaks, 2))) > 1, f"Artifact alert: identical spike positions detected across sites: {peaks}"


def main():
    manifest_csv = Path("data/manifest.csv")
    annotations_csv = Path("data/annotations.csv")

    if not manifest_csv.exists():
        logger.error("data/manifest.csv not found! Run scripts/build_manifest_and_cache.py first.")
        sys.exit(1)

    manifest_df = pd.read_csv(manifest_csv)
    ann_df = pd.read_csv(annotations_csv) if annotations_csv.exists() else pd.DataFrame()

    out_dir = Path("paper/figures/qc_grids")
    generate_site_qc_grids(manifest_df, ann_df, out_dir)
    generate_intensity_histograms(manifest_df, out_dir / "intensity_histograms.png")
    logger.info("Phase P1 QC grids and histograms generation complete!")


if __name__ == "__main__":
    main()
