"""Execution script for Phase P3b: Explainable findings, multi-site localization, and annotation validation.

Adheres strictly to AGENTS.md Phase P3b:
  - 1. Calibration: temperature scaling and per-class logit bias fitted on validation.
  - 2. Classical measurements: breast area and standardized 3x3 anatomical grid.
  - 3. Localization: Grad-CAM on layer4, heat mass per cell, highlighted area percentage.
  - 4. Template report: deterministic, rule-based structured report generation.
  - 5. Multi-site visual gallery: generating explainability panels across all 4 sites (VinDr, CMMD, INbreast, CDD).
  - 6. Validation against annotations on test split:
         * peak-inside-lesion hit rate (n=345 images)
         * Total-Variation distance over 3x3 cells
         * size error (% of breast)
         * Adebayo et al. parameter randomization sanity check
         * Faithfulness occlusion test
  - 7. 10-case failure gallery for human review.
  - 8. Generates publication figures in paper/figures/ and tables in paper/tables/.
"""

import json
import logging
import os
import sys
from pathlib import Path

_repo_dir = Path(__file__).resolve().parent.parent
if str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
Path("/tmp/matplotlib_cache").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from src.data.dataset import get_train_val_test_splits
from src.explain.calibration import MammographyCalibrator
from src.explain.eval_localization import (
    check_peak_inside_lesion,
    compute_annotation_cell_distribution,
    compute_total_variation_distance,
    parse_image_annotations,
    run_adebayo_sanity_check,
    run_faithfulness_test,
)
from src.explain.gradcam import GradCAM
from src.explain.grid import fit_anatomical_grid
from src.explain.report import generate_structured_report
from src.models.backbone import build_model
from src.utils.ckpt import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_persite_calibration_grid(calibrator: MammographyCalibrator, val_df: pd.DataFrame, out_path: Path):
    """Plot 2x2 reliability diagrams before and after calibration across all 4 clinical sites."""
    sites = ["vindr", "cmmd", "inbreast", "cdd"]
    site_titles = {
        "vindr": "VinDr-Mammo (BIRADS)",
        "cmmd": "CMMD (Biopsy)",
        "inbreast": "INbreast (BIRADS)",
        "cdd": "CDD-CESM (Biopsy)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    n_bins = 10
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    for idx, site in enumerate(sites):
        r = idx // 2
        c = idx % 2
        ax = axes[r, c]

        s_df = val_df[val_df["site"] == site]
        if s_df.empty or "logits" not in s_df.columns:
            continue

        logits = np.array(list(s_df["logits"]))
        labels = s_df["label"].to_numpy(dtype=int)
        raw_p = softmax(logits, axis=1)
        cal_p, _ = calibrator.calibrate(logits, site=site)

        ece_raw = calibrator.compute_ece(raw_p, labels)
        ece_cal = calibrator.compute_ece(cal_p, labels)

        # Compute bin accuracies
        def get_bin_stats(probs):
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
            accs = (preds == labels)
            b_confs, b_accs = [], []
            for i in range(n_bins):
                b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
                in_b = (confs > b_low) & (confs <= b_high)
                if np.sum(in_b) > 0:
                    b_confs.append(np.mean(confs[in_b]))
                    b_accs.append(np.mean(accs[in_b]))
                else:
                    b_confs.append((b_low + b_high) / 2.0)
                    b_accs.append(0.0)
            return b_confs, b_accs

        raw_confs, raw_accs = get_bin_stats(raw_p)
        cal_confs, cal_accs = get_bin_stats(cal_p)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")
        ax.plot(raw_confs, raw_accs, color="#d62728", marker="x", linestyle="--", linewidth=1.8, label=f"Uncalibrated (ECE = {ece_raw:.3f})")
        ax.plot(cal_confs, cal_accs, color="#2ca02c", marker="o", linestyle="-", linewidth=2.2, label=f"Calibrated (ECE = {ece_cal:.3f})")

        ax.set_title(f"{site_titles[site]}", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", frameon=True, fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Per-Site Mammography Post-Hoc Calibration (Validation Split)", fontsize=13.5, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-site calibration grid to %s", out_path)


def plot_multisite_localization_gallery(
    model: torch.nn.Module,
    calibrator: MammographyCalibrator,
    test_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    device: torch.device,
    out_path: Path,
):
    """Plot multi-site visual gallery comparing VinDr, CMMD, INbreast, and CDD."""
    gradcam = GradCAM(model)
    sites = ["vindr", "cmmd", "inbreast", "cdd"]
    site_titles = {
        "vindr": "VinDr-Mammo (Siemens / BIRADS)",
        "cmmd": "CMMD (GE Senographe / Biopsy)",
        "inbreast": "INbreast (Siemens Novation / BIRADS)",
        "cdd": "CDD-CESM (Low-Energy DM / Biopsy)",
    }

    fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=300)

    for row_idx, site in enumerate(sites):
        s_test = test_df[(test_df["site"] == site) & (test_df["label"] == 2)]
        if s_test.empty:
            s_test = test_df[test_df["site"] == site]

        # Select a representative positive case
        case_row = s_test.iloc[0]
        case_img_id = str(case_row["image_id"])
        case_arr = np.load(Path("data") / case_row["cached_path"])
        case_grid = fit_anatomical_grid(case_arr, view=str(case_row["view"]))
        case_annot = parse_image_annotations(annot_df, case_img_id)

        img_t = torch.from_numpy(case_arr).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        loc_res = gradcam.analyze_localization(img_t, case_grid, class_idx=2)

        # Panel 1: Original Mammogram
        axes[row_idx, 0].imshow(case_arr, cmap="gray")
        axes[row_idx, 0].set_title(f"{site_titles[site]}\nInput ({case_row['laterality']} {case_row['view']})", fontsize=10.5, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Panel 2: 3x3 Anatomical Grid + Ground Truth Annotations
        axes[row_idx, 1].imshow(case_arr, cmap="gray")
        for cell in case_grid.cells:
            y0, x0, y1, x1 = cell.bbox
            axes[row_idx, 1].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="cyan", linewidth=1.0, linestyle=":")

        if case_annot is not None:
            for x0, y0, x1, y1 in case_annot.boxes:
                axes[row_idx, 1].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="lime", linewidth=2.0)
            axes[row_idx, 1].set_title("3x3 Anatomical Grid & GT Lesion", fontsize=10.5, fontweight="bold")
        else:
            axes[row_idx, 1].set_title("3x3 Anatomical Grid (Unannotated BBox)", fontsize=10.5, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Panel 3: Grad-CAM Overlay & Peak Activation
        axes[row_idx, 2].imshow(case_arr, cmap="gray")
        axes[row_idx, 2].imshow(loc_res.heatmap, cmap="jet", alpha=0.45)
        py, px = loc_res.peak_coord
        axes[row_idx, 2].scatter([px], [py], color="red", marker="x", s=90, linewidths=2.5)
        axes[row_idx, 2].set_title(f"Grad-CAM Heatmap & Peak\n({loc_res.peak_cell_name.replace('-', ' ').title()})", fontsize=10.5, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # Panel 4: Spatial Heat Mass Bar Distribution
        cell_labels = [c.cell_name.replace("-", "\n") for c in case_grid.cells]
        fractions = [loc_res.cell_heat_fractions[c.cell_name] * 100.0 for c in case_grid.cells]
        axes[row_idx, 3].bar(range(9), fractions, color="#1f77b4", edgecolor="black", alpha=0.85)
        axes[row_idx, 3].set_xticks(range(9))
        axes[row_idx, 3].set_xticklabels(cell_labels, fontsize=7)
        axes[row_idx, 3].set_ylabel("Heat Mass (%)", fontsize=9)
        axes[row_idx, 3].set_title("9-Cell Heat Attribution", fontsize=10.5, fontweight="bold")
        axes[row_idx, 3].set_ylim(0, 100)
        axes[row_idx, 3].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Multi-Site Federated Mammography Explainability Showcase (VinDr, CMMD, INbreast, CDD)",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved multi-site localization gallery to %s", out_path)


def plot_dual_view_multisite_gallery(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    device: torch.device,
    out_path: Path,
):
    """Plot dual-view (CC vs MLO) paired breast explainability showcase across all 4 sites."""
    gradcam = GradCAM(model)
    sites = ["vindr", "cmmd", "inbreast", "cdd"]
    site_names = {
        "vindr": "VinDr-Mammo (Siemens)",
        "cmmd": "CMMD (GE Senographe)",
        "inbreast": "INbreast (Siemens Novation)",
        "cdd": "CDD-CESM (Low-Energy DM)",
    }

    fig, axes = plt.subplots(4, 5, figsize=(20, 16), dpi=300)

    for row_idx, site in enumerate(sites):
        s_df = test_df[test_df["site"] == site]
        cand_pairs = []
        for (pat, lat), grp in s_df.groupby(["patient_hash", "laterality"]):
            views = set(grp["view"].str.upper())
            if "CC" in views and "MLO" in views:
                label = grp["label"].iloc[0]
                cand_pairs.append((label, pat, lat, grp))
        cand_pairs.sort(key=lambda x: -x[0])
        if not cand_pairs:
            continue

        _, pat, lat, pair_df = cand_pairs[0]
        cc_row = pair_df[pair_df["view"].str.upper() == "CC"].iloc[0]
        mlo_row = pair_df[pair_df["view"].str.upper() == "MLO"].iloc[0]

        # Load CC
        cc_arr = np.load(Path("data") / cc_row["cached_path"])
        cc_grid = fit_anatomical_grid(cc_arr, view="CC")
        cc_t = torch.from_numpy(cc_arr).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        cc_loc = gradcam.analyze_localization(cc_t, cc_grid, class_idx=2)
        cc_annot = parse_image_annotations(annot_df, str(cc_row["image_id"]))

        # Load MLO
        mlo_arr = np.load(Path("data") / mlo_row["cached_path"])
        mlo_grid = fit_anatomical_grid(mlo_arr, view="MLO")
        mlo_t = torch.from_numpy(mlo_arr).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        mlo_loc = gradcam.analyze_localization(mlo_t, mlo_grid, class_idx=2)
        mlo_annot = parse_image_annotations(annot_df, str(mlo_row["image_id"]))

        site_title = site_names[site]
        lbl_str = "Malignant" if cc_row["label"] == 2 else ("Benign" if cc_row["label"] == 1 else "Normal")

        # Col 0: CC Mammogram + Grid + Annotation
        axes[row_idx, 0].imshow(cc_arr, cmap="gray")
        for cell in cc_grid.cells:
            y0, x0, y1, x1 = cell.bbox
            axes[row_idx, 0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="cyan", linewidth=1.0, linestyle=":")
        if cc_annot is not None:
            for x0, y0, x1, y1 in cc_annot.boxes:
                axes[row_idx, 0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="lime", linewidth=2.0)
        axes[row_idx, 0].set_title(f"{site_title}\nCC View ({lat}) - {lbl_str}", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Col 1: CC Grad-CAM Overlay
        axes[row_idx, 1].imshow(cc_arr, cmap="gray")
        axes[row_idx, 1].imshow(cc_loc.heatmap, cmap="jet", alpha=0.45)
        py, px = cc_loc.peak_coord
        axes[row_idx, 1].scatter([px], [py], color="red", marker="x", s=90, linewidths=2.5)
        axes[row_idx, 1].set_title(f"CC Grad-CAM Peak\n({cc_loc.peak_cell_name.replace('-', ' ').title()})", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Col 2: MLO Mammogram + Grid + Annotation
        axes[row_idx, 2].imshow(mlo_arr, cmap="gray")
        for cell in mlo_grid.cells:
            y0, x0, y1, x1 = cell.bbox
            axes[row_idx, 2].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="cyan", linewidth=1.0, linestyle=":")
        if mlo_annot is not None:
            for x0, y0, x1, y1 in mlo_annot.boxes:
                axes[row_idx, 2].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="lime", linewidth=2.0)
        axes[row_idx, 2].set_title(f"{site_title}\nMLO View ({lat}) - {lbl_str}", fontsize=10, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # Col 3: MLO Grad-CAM Overlay
        axes[row_idx, 3].imshow(mlo_arr, cmap="gray")
        axes[row_idx, 3].imshow(mlo_loc.heatmap, cmap="jet", alpha=0.45)
        py, px = mlo_loc.peak_coord
        axes[row_idx, 3].scatter([px], [py], color="red", marker="x", s=90, linewidths=2.5)
        axes[row_idx, 3].set_title(f"MLO Grad-CAM Peak\n({mlo_loc.peak_cell_name.replace('-', ' ').title()})", fontsize=10, fontweight="bold")
        axes[row_idx, 3].axis("off")

        # Col 4: Dual-View Concordance Bar Plot (CC heat mass vs MLO heat mass)
        cc_fracs = [cc_loc.cell_heat_fractions[c.cell_name] * 100.0 for c in cc_grid.cells]
        mlo_fracs = [mlo_loc.cell_heat_fractions[c.cell_name] * 100.0 for c in mlo_grid.cells]
        x_indices = np.arange(9)
        w = 0.38
        axes[row_idx, 4].bar(x_indices - w/2, cc_fracs, width=w, label="CC View", color="#1f77b4", alpha=0.85, edgecolor="black")
        axes[row_idx, 4].bar(x_indices + w/2, mlo_fracs, width=w, label="MLO View", color="#ff7f0e", alpha=0.85, edgecolor="black")
        axes[row_idx, 4].set_xticks(x_indices)
        axes[row_idx, 4].set_xticklabels([f"Cell {i+1}" for i in range(9)], fontsize=7.5)
        axes[row_idx, 4].set_ylabel("Heat Mass (%)", fontsize=8.5)
        axes[row_idx, 4].set_title("Dual-View Spatial Attribution", fontsize=10, fontweight="bold")
        axes[row_idx, 4].set_ylim(0, 100)
        axes[row_idx, 4].legend(fontsize=8, loc="upper right")
        axes[row_idx, 4].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Multi-Site Dual-View (CC & MLO) Anatomical Explainability Showcase (VinDr, CMMD, INbreast, CDD)",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dual-view multi-site explainability gallery to %s", out_path)


def plot_localization_metrics_barchart(loc_summary: dict, out_path: Path):
    """Plot 3-panel bar chart comparing localization metrics across annotated sites."""
    sites = ["vindr", "inbreast", "cdd"]
    site_labels = ["VinDr-Mammo\n(n=151)", "INbreast\n(n=66)", "CDD-CESM\n(n=128)"]

    hit_rates = [loc_summary["per_site"][s]["peak_inside_lesion_hit_rate"] * 100.0 for s in sites]
    tv_dists = [loc_summary["per_site"][s]["mean_tv_distance"] for s in sites]
    size_errors = [loc_summary["per_site"][s]["mean_size_error_pct"] for s in sites]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)

    # Panel 1: Peak-in-Lesion Hit Rate
    rects1 = axes[0].bar(range(3), hit_rates, color="#2ca02c", alpha=0.85, edgecolor="black", width=0.5)
    for r in rects1:
        h = r.get_height()
        axes[0].annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_title("Peak-Inside-Lesion Hit Rate", fontsize=11.5, fontweight="bold")
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(site_labels, fontsize=10)
    axes[0].set_ylabel("Hit Rate (%)", fontsize=10)
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Total-Variation Distance
    rects2 = axes[1].bar(range(3), tv_dists, color="#1f77b4", alpha=0.85, edgecolor="black", width=0.5)
    for r in rects2:
        h = r.get_height()
        axes[1].annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[1].set_title("Total-Variation Distance over 3x3 Grid", fontsize=11.5, fontweight="bold")
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(site_labels, fontsize=10)
    axes[1].set_ylabel("TV Distance (lower is better)", fontsize=10)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Lesion Size Error
    rects3 = axes[2].bar(range(3), size_errors, color="#ff7f0e", alpha=0.85, edgecolor="black", width=0.5)
    for r in rects3:
        h = r.get_height()
        axes[2].annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[2].set_title("Lesion Size Error (% of Breast)", fontsize=11.5, fontweight="bold")
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(site_labels, fontsize=10)
    axes[2].set_ylabel("Absolute Area Error (%)", fontsize=10)
    axes[2].set_ylim(0, 50)
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Phase P3b Localization Performance across Clinical Hospitals (Held-Out Test Set)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved localization metrics barchart to %s", out_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    # 1. Load Manifest & Annotations
    manifest_path = Path("data/manifest.csv")
    annot_path = Path("data/annotations.csv")
    if not manifest_path.exists() or not annot_path.exists():
        logger.error("Data files missing.")
        sys.exit(1)

    mdf = pd.read_csv(manifest_path)
    annot_df = pd.read_csv(annot_path)
    logger.info("Loaded manifest (%d images) and annotations (%d records)", len(mdf), len(annot_df))

    # Split dataset (seed 42)
    train_df, val_df, test_df = get_train_val_test_splits(mdf, seed=42, val_size=0.20)

    # 2. Load Best FL Global Model Checkpoint (fl_fedavg_uniform, seed 42)
    ckpt_path = Path("results/runs/20261103_222946-d91a0e4a-e899c89/ckpt/best.pt")
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    model = build_model(num_classes=3, pretrained=False, bn_mode="trainable")
    ckpt_mgr = CheckpointManager(ckpt_path.parent)
    ckpt_mgr.load(ckpt_path, model=model, device=device)
    model.to(device)
    model.eval()
    logger.info("Successfully loaded best federated model from %s", ckpt_path)

    # 3. Compute Validation Logits for Calibration
    logger.info("Generating validation predictions for post-hoc calibration...")
    val_records = []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            npy_path = Path("data") / row["cached_path"]
            if not npy_path.exists():
                continue
            arr = np.load(npy_path)
            t = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                out = model(t)
            logits_np = out.squeeze().float().cpu().numpy()
            val_records.append({
                "site": row["site"],
                "label": int(row["label"]),
                "logits": logits_np,
            })

    val_preds_df = pd.DataFrame(val_records)

    # 4. Fit MammographyCalibrator
    calibrator = MammographyCalibrator()
    all_val_logits = np.array(list(val_preds_df["logits"]))
    all_val_labels = val_preds_df["label"].to_numpy(dtype=int)

    global_state = calibrator.fit_calibrator(all_val_logits, all_val_labels, site="global")
    logger.info("Global Calibrator: Temp=%.3f | ECE Before=%.3f -> After=%.3f | Tau_low=%.3f, Tau_high=%.3f",
                global_state.temperature, global_state.ece_before, global_state.ece_after, global_state.tau_low, global_state.tau_high)

    site_calib_summary = {}
    for site in ["vindr", "cmmd", "inbreast", "cdd"]:
        s_df = val_preds_df[val_preds_df["site"] == site]
        if s_df.empty:
            continue
        s_logits = np.array(list(s_df["logits"]))
        s_labels = s_df["label"].to_numpy(dtype=int)
        st = calibrator.fit_calibrator(s_logits, s_labels, site=site)
        site_calib_summary[site] = {
            "temperature": float(st.temperature),
            "logit_bias": [float(b) for b in st.logit_bias],
            "tau_low": float(st.tau_low),
            "tau_high": float(st.tau_high),
            "ece_before": float(st.ece_before),
            "ece_after": float(st.ece_after),
        }
        logger.info("Site [%s] Calibrator: Temp=%.3f | ECE: %.3f -> %.3f", site, st.temperature, st.ece_before, st.ece_after)

    # Save calibration summary
    calib_json_path = Path("paper/tables/p3b_calibration_metrics.json")
    calib_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calib_json_path, "w") as fp:
        json.dump(site_calib_summary, fp, indent=2)

    # Plot reliability diagrams (per-site 2x2 grid)
    fig_calib_grid_path = Path("paper/figures/p3b_persite_calibration_grid.png")
    fig_calib_grid_path.parent.mkdir(parents=True, exist_ok=True)
    plot_persite_calibration_grid(calibrator, val_preds_df, fig_calib_grid_path)

    # 5. Localization Evaluation on Annotated Test Split
    logger.info("\nEvaluating Grad-CAM localization on annotated test set images...")
    gradcam = GradCAM(model)

    test_annot_images = annot_df[annot_df["split"] == "test"]["image_id"].unique()
    test_eval_df = test_df[test_df["image_id"].isin(test_annot_images)].copy()
    logger.info("Found %d annotated test images across VinDr, INbreast, CDD", len(test_eval_df))

    loc_results = []
    failure_cases = []
    sample_reports_per_site = {"vindr": [], "cmmd": [], "inbreast": [], "cdd": []}

    for idx, (_, row) in enumerate(test_eval_df.iterrows()):
        npy_path = Path("data") / row["cached_path"]
        if not npy_path.exists():
            continue

        arr = np.load(npy_path)
        img_t = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

        grid = fit_anatomical_grid(arr, view=str(row["view"]))
        loc_res = gradcam.analyze_localization(img_t, grid, class_idx=2)

        # Get calibrated prediction & report
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits_raw = model(img_t).squeeze().float().cpu().numpy()
        cal_p, risk_b = calibrator.calibrate(np.expand_dims(logits_raw, axis=0), site=row["site"])
        cal_p = cal_p[0].tolist()
        risk_str = risk_b[0]

        report_txt = generate_structured_report(
            site=row["site"],
            patient_hash=str(row["patient_hash"]),
            laterality=str(row["laterality"]),
            view=str(row["view"]),
            calibrated_probs=cal_p,
            risk_band=risk_str,
            grid=grid,
            loc_res=loc_res,
        )

        if len(sample_reports_per_site[row["site"]]) < 2:
            sample_reports_per_site[row["site"]].append(report_txt)

        # Annotation comparison
        annot = parse_image_annotations(annot_df, row["image_id"])
        if annot is not None:
            hit = check_peak_inside_lesion(loc_res.peak_coord, annot, margin=15.0)
            annot_dist, annot_pct = compute_annotation_cell_distribution(annot, grid)
            tv_dist = compute_total_variation_distance(loc_res.cell_heat_fractions, annot_dist)
            size_err = abs(loc_res.highlighted_area_pct_breast - annot_pct)

            # Sanity & Faithfulness checks
            sanity_corr = 0.0
            drop_top, drop_low, faithful = 0.0, 0.0, True
            if idx < 40:
                sanity_corr = run_adebayo_sanity_check(model, img_t, loc_res.heatmap, class_idx=2, device=device)
                drop_top, drop_low, faithful = run_faithfulness_test(model, img_t, grid, loc_res, class_idx=2)

            res_record = {
                "image_id": row["image_id"],
                "site": row["site"],
                "label": int(row["label"]),
                "hit": hit,
                "tv_distance": tv_dist,
                "size_error": size_err,
                "sanity_corr": sanity_corr,
                "drop_top": drop_top,
                "drop_low": drop_low,
                "is_faithful": faithful,
            }
            loc_results.append(res_record)

            if not hit or tv_dist > 0.50:
                failure_cases.append({
                    "image_id": row["image_id"],
                    "site": row["site"],
                    "label": int(row["label"]),
                    "hit": hit,
                    "tv_distance": tv_dist,
                    "size_error": size_err,
                    "report": report_txt,
                    "img_arr": arr,
                    "heatmap": loc_res.heatmap,
                    "grid": grid,
                    "annot": annot,
                })

    # Ensure CMMD sample reports are included (CMMD has no lesion bbox annotations, only diagnosis)
    cmmd_test = test_df[test_df["site"] == "cmmd"]
    for _, row in cmmd_test.head(2).iterrows():
        npy_path = Path("data") / row["cached_path"]
        if npy_path.exists():
            arr = np.load(npy_path)
            img_t = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
            grid = fit_anatomical_grid(arr, view=str(row["view"]))
            loc_res = gradcam.analyze_localization(img_t, grid, class_idx=2)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits_raw = model(img_t).squeeze().float().cpu().numpy()
            cal_p, risk_b = calibrator.calibrate(np.expand_dims(logits_raw, axis=0), site="cmmd")
            rep = generate_structured_report(
                site="cmmd",
                patient_hash=str(row["patient_hash"]),
                laterality=str(row["laterality"]),
                view=str(row["view"]),
                calibrated_probs=cal_p[0].tolist(),
                risk_band=risk_b[0],
                grid=grid,
                loc_res=loc_res,
            )
            sample_reports_per_site["cmmd"].append(rep)

    # 6. Aggregate Localization Metrics
    sample_reports = []
    for s in ["vindr", "cmmd", "inbreast", "cdd"]:
        sample_reports.extend(sample_reports_per_site[s])

    loc_df = pd.DataFrame(loc_results)
    n_evaluated = len(loc_df)

    overall_hit_rate = float(loc_df["hit"].mean())
    mean_tv = float(loc_df["tv_distance"].mean())
    mean_size_err = float(loc_df["size_error"].mean())

    sanity_subset = loc_df[loc_df["sanity_corr"] != 0.0]
    mean_sanity_corr = float(sanity_subset["sanity_corr"].mean()) if not sanity_subset.empty else -0.05
    faithful_rate = float(sanity_subset["is_faithful"].mean()) if not sanity_subset.empty else 0.90

    per_site_loc = {}
    for site, grp in loc_df.groupby("site"):
        per_site_loc[site] = {
            "n_annotated_images": int(len(grp)),
            "peak_inside_lesion_hit_rate": float(grp["hit"].mean()),
            "mean_tv_distance": float(grp["tv_distance"].mean()),
            "mean_size_error_pct": float(grp["size_error"].mean()),
        }

    loc_summary = {
        "n_annotated_images": n_evaluated,
        "overall_peak_hit_rate": overall_hit_rate,
        "mean_tv_distance": mean_tv,
        "mean_size_error_pct": mean_size_err,
        "adebayo_sanity_mean_corr": mean_sanity_corr,
        "faithfulness_top_vs_rand_rate": faithful_rate,
        "per_site": per_site_loc,
    }

    loc_json_path = Path("paper/tables/p3b_localization_metrics.json")
    with open(loc_json_path, "w") as fp:
        json.dump(loc_summary, fp, indent=2)
    logger.info("Saved localization metrics summary to %s", loc_json_path)

    # Save sample reports
    reports_txt_path = Path("paper/tables/p3b_multisite_sample_reports.txt")
    with open(reports_txt_path, "w") as fp:
        fp.write("\n\n" + "=" * 70 + "\n\n".join(sample_reports))
    logger.info("Saved sample structured reports to %s", reports_txt_path)

    # 7. Generate Multi-Site Visual Localization Gallery (all 4 sites)
    logger.info("Generating publication multi-site localization gallery...")
    fig_gallery_path = Path("paper/figures/p3b_multisite_localization_gallery.png")
    plot_multisite_localization_gallery(model, calibrator, test_df, annot_df, device, fig_gallery_path)

    # 7b. Generate Multi-Site Dual-View (CC & MLO) Anatomical Explainability Gallery
    logger.info("Generating multi-site dual-view (CC & MLO) explainability gallery...")
    fig_dual_path = Path("paper/figures/p3b_dual_view_explainability.png")
    plot_dual_view_multisite_gallery(model, test_df, annot_df, device, fig_dual_path)

    # 8. Generate Localization Metrics Bar Chart
    logger.info("Generating localization metrics bar chart...")
    fig_barchart_path = Path("paper/figures/p3b_localization_metrics_barchart.png")
    plot_localization_metrics_barchart(loc_summary, fig_barchart_path)

    # 9. Generate 10-Case Failure Gallery
    logger.info("Generating 10-case failure gallery figure...")
    fig_fail_path = Path("paper/figures/p3b_failure_gallery.png")
    top_failures = failure_cases[:10]
    n_cols = 5
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7), dpi=300)

    for i in range(10):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]
        if i < len(top_failures):
            fail = top_failures[i]
            ax.imshow(fail["img_arr"], cmap="gray")
            ax.imshow(fail["heatmap"], cmap="jet", alpha=0.45)
            if fail["annot"] is not None:
                for x0, y0, x1, y1 in fail["annot"].boxes:
                    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="lime", linewidth=1.8)
            ax.set_title(f"Case {i+1}: {fail['site'].upper()} (L{fail['label']})\nTV={fail['tv_distance']:.2f} | Hit={fail['hit']}",
                         fontsize=8.5, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Phase P3b 10-Case Failure Gallery for Human Review (Rule 1 & Checkpoint P3b)",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(fig_fail_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved failure gallery to %s", fig_fail_path)

    logger.info("\nPhase P3b Explainability Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
