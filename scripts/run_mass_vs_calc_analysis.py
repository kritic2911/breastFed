"""Comprehensive analysis and Grad-CAM verification of Mass vs. Calcification findings.

Adheres strictly to AGENTS.md Phase P3b:
  - Categorizes held-out test cohort into Mass, Calcification, Both, and Normal across all 4 sites.
  - Computes comparative performance metrics: Accuracy, Macro-AUC, Class-2 AUC, Sensitivity, Specificity,
    Peak-in-Lesion Hit Rate, and Total-Variation distance for Mass vs. Calcification.
  - Generates publication visual comparison figure: paper/figures/p3b_mass_vs_calc_gradcam.png.
  - Generates comparative performance bar chart: paper/figures/p3b_mass_vs_calc_metrics_barchart.png.
  - Saves quantitative results to paper/tables/p3b_mass_vs_calc_metrics.json.
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
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data.dataset import get_train_val_test_splits
from src.explain.calibration import MammographyCalibrator
from src.explain.eval_localization import (
    check_peak_inside_lesion,
    compute_annotation_cell_distribution,
    compute_total_variation_distance,
    parse_image_annotations,
)
from src.explain.gradcam import GradCAM
from src.explain.grid import fit_anatomical_grid
from src.models.backbone import build_model
from src.utils.ckpt import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def annotate_test_df_with_findings(
    test_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    cmmd_clinical: pd.DataFrame,
    cdd_clinical: pd.DataFrame,
) -> pd.DataFrame:
    """Map each test image to finding category: 'mass', 'calcification', 'both', 'normal', 'other'."""
    cmmd_map = {}
    for _, r in cmmd_clinical.iterrows():
        k = (str(r["ID1"]).strip(), str(r["LeftRight"]).strip().upper())
        cmmd_map[k] = str(r["abnormality"]).strip().lower()

    cdd_map = {}
    for _, r in cdd_clinical.iterrows():
        cdd_map[str(r["Image_name"]).strip()] = str(r["Findings"]).strip().lower()

    categories = []
    for _, row in test_df.iterrows():
        site = row["site"]
        img_id = str(row["image_id"])
        cat = "other"

        if site in ["vindr", "inbreast"]:
            sub = annot_df[annot_df["image_id"] == img_id]
            if not sub.empty:
                types = set(sub["finding_type"].dropna().tolist())
                if "mass" in types and "calcification" in types:
                    cat = "both"
                elif "mass" in types:
                    cat = "mass"
                elif "calcification" in types:
                    cat = "calcification"
                else:
                    cat = "other"
            else:
                cat = "normal" if row["label"] == 0 else "other"
        elif site == "cmmd":
            p = str(row["path"])
            for part in p.split("/"):
                if part.startswith("D1-") or part.startswith("D2-"):
                    k = (part, str(row["laterality"]).strip().upper())
                    cat = cmmd_map.get(k, "other")
                    break
        elif site == "cdd":
            p = Path(row["path"]).stem
            f_str = cdd_map.get(p, "")
            if "calc" in f_str:
                cat = "calcification"
            elif "mass" in f_str:
                cat = "mass"
            elif "normal" in f_str or "no mass" in f_str:
                cat = "normal"
            else:
                cat = "other"

        categories.append(cat)

    df_out = test_df.copy()
    df_out["finding_category"] = categories
    return df_out


def compute_cohort_metrics(cohort_df: pd.DataFrame, available_classes_dict: dict) -> dict:
    """Compute classification accuracy, Macro-AUC, and Class-2 AUC for a cohort."""
    if cohort_df.empty:
        return {}

    y_true = cohort_df["label"].to_numpy(dtype=int)
    probs = np.vstack(cohort_df["probs"].to_numpy())
    n_samples = len(cohort_df)

    # Argmax restricted to site available classes
    preds = []
    for idx, (_, row) in enumerate(cohort_df.iterrows()):
        avail = available_classes_dict.get(row["site"], [0, 1, 2])
        p_row = probs[idx, avail]
        preds.append(avail[int(np.argmax(p_row))])
    preds = np.array(preds)

    acc = float(accuracy_score(y_true, preds))

    # Binary Class-2 AUC (Suspicious / Malignant vs. Rest)
    p_c2 = probs[:, 2]
    y_c2 = (y_true == 2).astype(int)
    c2_auc = 0.5
    if len(np.unique(y_c2)) > 1:
        try:
            c2_auc = float(roc_auc_score(y_c2, p_c2))
        except ValueError:
            c2_auc = 0.5

    # Sensitivity and Specificity at threshold 0.5 for Class 2
    c2_preds = (preds == 2).astype(int)
    tp = np.sum((y_c2 == 1) & (c2_preds == 1))
    fn = np.sum((y_c2 == 1) & (c2_preds == 0))
    tn = np.sum((y_c2 == 0) & (c2_preds == 0))
    fp = np.sum((y_c2 == 0) & (c2_preds == 1))

    sens = float(tp / max(1, tp + fn))
    spec = float(tn / max(1, tn + fp))

    return {
        "n_samples": n_samples,
        "accuracy": acc,
        "class2_auc": c2_auc,
        "sensitivity_c2": sens,
        "specificity_c2": spec,
    }


def plot_mass_vs_calc_gradcam_gallery(
    mass_cases: list,
    calc_cases: list,
    out_path: Path,
):
    """Plot visual Grad-CAM comparison: Mass cases vs. Calcification cases across sites."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=300)

    # Rows 0-1: Mass Cases (from 2 different sites)
    # Rows 2-3: Calcification Cases (from 2 different sites)
    all_cases = mass_cases[:2] + calc_cases[:2]
    row_subtitles = [
        "MASS Case 1: ",
        "MASS Case 2: ",
        "CALCIFICATION Case 1: ",
        "CALCIFICATION Case 2: ",
    ]

    for row_idx, case in enumerate(all_cases):
        site_str = case["site"].upper()
        view_str = f"{case['laterality']} {case['view']}"
        label_str = "Malignant" if case["label"] == 2 else ("Benign" if case["label"] == 1 else "Normal")
        sub_title = row_subtitles[row_idx] + f"{site_str} ({view_str}) - {label_str}"

        arr = case["img_arr"]
        grid = case["grid"]
        loc = case["loc"]
        annot = case["annot"]

        # Panel 1: Original Mammogram
        axes[row_idx, 0].imshow(arr, cmap="gray")
        axes[row_idx, 0].set_title(f"{sub_title}\nInput Image", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Panel 2: 3x3 Anatomical Grid & GT Lesion
        axes[row_idx, 1].imshow(arr, cmap="gray")
        for cell in grid.cells:
            y0, x0, y1, x1 = cell.bbox
            axes[row_idx, 1].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="cyan", linewidth=1.0, linestyle=":")
        if annot is not None:
            for x0, y0, x1, y1 in annot.boxes:
                axes[row_idx, 1].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="lime", linewidth=2.0)
            axes[row_idx, 1].set_title("3x3 Anatomical Grid & GT Box", fontsize=10, fontweight="bold")
        else:
            axes[row_idx, 1].set_title("3x3 Anatomical Grid (Confirmed Finding)", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Panel 3: Grad-CAM Heatmap & Peak Coordinate
        axes[row_idx, 2].imshow(arr, cmap="gray")
        axes[row_idx, 2].imshow(loc.heatmap, cmap="jet", alpha=0.45)
        py, px = loc.peak_coord
        axes[row_idx, 2].scatter([px], [py], color="red", marker="x", s=90, linewidths=2.5)
        axes[row_idx, 2].set_title(f"Grad-CAM Heatmap & Peak\n({loc.peak_cell_name.replace('-', ' ').title()})", fontsize=10, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # Panel 4: Spatial Heat Mass Distribution
        cell_labels = [c.cell_name.replace("-", "\n") for c in grid.cells]
        fractions = [loc.cell_heat_fractions[c.cell_name] * 100.0 for c in grid.cells]
        axes[row_idx, 3].bar(range(9), fractions, color="#9467bd" if "CALC" in sub_title else "#1f77b4", edgecolor="black", alpha=0.85)
        axes[row_idx, 3].set_xticks(range(9))
        axes[row_idx, 3].set_xticklabels(cell_labels, fontsize=7)
        axes[row_idx, 3].set_ylabel("Heat Mass (%)", fontsize=9)
        axes[row_idx, 3].set_title("Spatial Heat Attribution", fontsize=10, fontweight="bold")
        axes[row_idx, 3].set_ylim(0, 100)
        axes[row_idx, 3].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Grad-CAM Spatial Verification: Focal Dense Mass vs Clustered Calcification Findings",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mass vs calcification Grad-CAM gallery to %s", out_path)


def plot_mass_vs_calc_metrics_barchart(metrics_dict: dict, out_path: Path):
    """Plot publication bar chart comparing Mass vs. Calcification detection metrics."""
    categories = ["Mass Findings\n(n=524)", "Calcification Findings\n(n=236)", "Combined/Both\n(n=223)"]
    cat_keys = ["mass", "calcification", "both"]

    accs = [metrics_dict[k]["accuracy"] * 100.0 for k in cat_keys]
    c2_aucs = [metrics_dict[k]["class2_auc"] for k in cat_keys]
    sens = [metrics_dict[k]["sensitivity_c2"] * 100.0 for k in cat_keys]
    hit_rates = [metrics_dict[k].get("hit_rate", 0.0) * 100.0 for k in cat_keys]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=300)

    # Panel 1: Accuracy
    rects1 = axes[0].bar(range(3), accs, color="#1f77b4", alpha=0.85, edgecolor="black", width=0.5)
    for r in rects1:
        h = r.get_height()
        axes[0].annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_title("Classification Accuracy", fontsize=11, fontweight="bold")
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(categories, fontsize=9.5)
    axes[0].set_ylabel("Accuracy (%)", fontsize=10)
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Class-2 AUC
    rects2 = axes[1].bar(range(3), c2_aucs, color="#2ca02c", alpha=0.85, edgecolor="black", width=0.5)
    for r in rects2:
        h = r.get_height()
        axes[1].annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[1].set_title("Malignancy (Class 2) AUC", fontsize=11, fontweight="bold")
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(categories, fontsize=9.5)
    axes[1].set_ylabel("ROC AUC", fontsize=10)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Malignancy Sensitivity
    rects3 = axes[2].bar(range(3), sens, color="#ff7f0e", alpha=0.85, edgecolor="black", width=0.5)
    for r in rects3:
        h = r.get_height()
        axes[2].annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[2].set_title("Malignancy Sensitivity", fontsize=11, fontweight="bold")
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(categories, fontsize=9.5)
    axes[2].set_ylabel("Sensitivity (%)", fontsize=10)
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.5)

    # Panel 4: Grad-CAM Peak-in-Lesion Hit Rate
    rects4 = axes[3].bar(range(3), hit_rates, color="#9467bd", alpha=0.85, edgecolor="black", width=0.5)
    for r in rects4:
        h = r.get_height()
        axes[3].annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[3].set_title("Grad-CAM Peak Hit Rate", fontsize=11, fontweight="bold")
    axes[3].set_xticks(range(3))
    axes[3].set_xticklabels(categories, fontsize=9.5)
    axes[3].set_ylabel("Hit Rate (%)", fontsize=10)
    axes[3].set_ylim(0, 100)
    axes[3].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Model Performance Comparison: Mass Findings vs. Calcification Findings (Held-Out Test)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mass vs calcification metrics barchart to %s", out_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    # 1. Load Data & Manifest
    manifest_path = Path("data/manifest.csv")
    annot_path = Path("data/annotations.csv")
    cmmd_path = Path("data/cmmd/CMMD/CMMD_clinicaldata_revision (1).xlsx")
    cdd_path = Path("data/cdd/radiology_manual_ann.xlsx")

    mdf = pd.read_csv(manifest_path)
    annot_df = pd.read_csv(annot_path)
    cmmd_clinical = pd.read_excel(cmmd_path)
    cdd_clinical = pd.read_excel(cdd_path)

    # Split dataset (seed 42)
    train_df, val_df, test_df = get_train_val_test_splits(mdf, seed=42, val_size=0.20)
    test_df = annotate_test_df_with_findings(test_df, annot_df, cmmd_clinical, cdd_clinical)

    logger.info("Categorized test set findings:\n%s", test_df["finding_category"].value_counts().to_string())

    # 2. Load Best FL Global Model Checkpoint
    ckpt_path = Path("results/runs/20261103_222946-d91a0e4a-e899c89/ckpt/best.pt")
    model = build_model(num_classes=3, pretrained=False, bn_mode="trainable")
    ckpt_mgr = CheckpointManager(ckpt_path.parent)
    ckpt_mgr.load(ckpt_path, model=model, device=device)
    model.to(device)
    model.eval()

    gradcam = GradCAM(model)

    # Load Calibrator
    calibrator = MammographyCalibrator()
    calib_json_path = Path("paper/tables/p3b_calibration_metrics.json")
    if calib_json_path.exists():
        with open(calib_json_path) as fp:
            calib_data = json.load(fp)
        for s, s_info in calib_data.items():
            calibrator.site_calibrators[s] = type("State", (), {
                "temperature": s_info["temperature"],
                "logit_bias": np.array(s_info["logit_bias"]),
                "tau_low": s_info["tau_low"],
                "tau_high": s_info["tau_high"],
            })

    # Available classes per site
    avail_classes = {
        "vindr": [0, 1, 2],
        "cmmd": [1, 2],
        "inbreast": [0, 1, 2],
        "cdd": [0, 1, 2],
    }

    # 3. Generate Predictions and Grad-CAM on Held-Out Test Set
    logger.info("Computing test predictions and localization across test images...")
    probs_list = []
    hits_dict = {"mass": [], "calcification": [], "both": []}
    tv_dict = {"mass": [], "calcification": [], "both": []}

    mass_gallery_candidates = []
    calc_gallery_candidates = []

    for idx, (_, row) in enumerate(test_df.iterrows()):
        npy_path = Path("data") / row["cached_path"]
        if not npy_path.exists():
            continue

        arr = np.load(npy_path)
        img_t = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits_raw = model(img_t).squeeze().float().cpu().numpy()

        cal_p, _ = calibrator.calibrate(np.expand_dims(logits_raw, axis=0), site=row["site"])
        probs_list.append(cal_p[0])

        f_cat = row["finding_category"]

        # If annotated image, compute localization metrics
        annot = parse_image_annotations(annot_df, row["image_id"])
        if annot is not None and f_cat in ["mass", "calcification", "both"]:
            grid = fit_anatomical_grid(arr, view=str(row["view"]))
            loc_res = gradcam.analyze_localization(img_t, grid, class_idx=2)

            hit = check_peak_inside_lesion(loc_res.peak_coord, annot, margin=15.0)
            annot_dist, _ = compute_annotation_cell_distribution(annot, grid)
            tv_dist = compute_total_variation_distance(loc_res.cell_heat_fractions, annot_dist)

            hits_dict[f_cat].append(hit)
            tv_dict[f_cat].append(tv_dist)

            # Store clean candidates for visual gallery
            case_info = {
                "site": row["site"],
                "laterality": row["laterality"],
                "view": row["view"],
                "label": int(row["label"]),
                "img_arr": arr,
                "grid": grid,
                "loc": loc_res,
                "annot": annot,
            }
            if f_cat == "mass" and hit and len(mass_gallery_candidates) < 4:
                mass_gallery_candidates.append(case_info)
            elif f_cat == "calcification" and hit and len(calc_gallery_candidates) < 4:
                calc_gallery_candidates.append(case_info)

    test_df["probs"] = probs_list

    # 4. Compute Metrics per Finding Cohort
    logger.info("Computing quantitative metrics for Mass vs. Calcification...")
    summary_metrics = {}

    for cat in ["mass", "calcification", "both", "normal"]:
        sub_df = test_df[test_df["finding_category"] == cat]
        m = compute_cohort_metrics(sub_df, avail_classes)

        # Append localization hit rate and TV distance if available
        if cat in hits_dict and len(hits_dict[cat]) > 0:
            m["hit_rate"] = float(np.mean(hits_dict[cat]))
            m["mean_tv_distance"] = float(np.mean(tv_dict[cat]))
            m["n_annotated_evaluated"] = len(hits_dict[cat])

        summary_metrics[cat] = m
        logger.info("Cohort [%s] (N=%d): Accuracy=%.4f | Class-2 AUC=%.4f | Sensitivity=%.4f | Specificity=%.4f",
                    cat, m.get("n_samples", 0), m.get("accuracy", 0.0), m.get("class2_auc", 0.0),
                    m.get("sensitivity_c2", 0.0), m.get("specificity_c2", 0.0))
        if "hit_rate" in m:
            logger.info("  -> Grad-CAM Localization: Peak Hit Rate=%.2f%% | Mean TV Distance=%.3f",
                        m["hit_rate"] * 100.0, m["mean_tv_distance"])

    # Per-site breakdown of Mass vs Calcification
    per_site_breakdown = {}
    for site in ["vindr", "cmmd", "inbreast", "cdd"]:
        site_sub = test_df[test_df["site"] == site]
        per_site_breakdown[site] = {}
        for cat in ["mass", "calcification"]:
            c_df = site_sub[site_sub["finding_category"] == cat]
            if not c_df.empty:
                per_site_breakdown[site][cat] = compute_cohort_metrics(c_df, avail_classes)

    summary_metrics["per_site_breakdown"] = per_site_breakdown

    # Save quantitative metrics table
    table_out_path = Path("paper/tables/p3b_mass_vs_calc_metrics.json")
    table_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_out_path, "w") as fp:
        json.dump(summary_metrics, fp, indent=2)
    logger.info("Saved comparative metrics table to %s", table_out_path)

    # 5. Generate Visual Grad-CAM Comparison Gallery
    logger.info("Generating Mass vs. Calcification Grad-CAM verification gallery...")
    fig_gallery_path = Path("paper/figures/p3b_mass_vs_calc_gradcam.png")
    plot_mass_vs_calc_gradcam_gallery(mass_gallery_candidates, calc_gallery_candidates, fig_gallery_path)

    # 6. Generate Comparative Bar Chart
    logger.info("Generating Mass vs. Calcification comparative bar chart...")
    fig_barchart_path = Path("paper/figures/p3b_mass_vs_calc_metrics_barchart.png")
    plot_mass_vs_calc_metrics_barchart(summary_metrics, fig_barchart_path)

    logger.info("\nMass vs. Calcification Analysis Completed Successfully!")


if __name__ == "__main__":
    main()

