"""Empirical evaluation of localization sharpening variants on the calcification test cohort.

Adheres strictly to AGENTS.md Phase P3b, Rule 9 Geometry, and Decision D-010:
  - Compares:
      1. Pure Grad-CAM (baseline coarse 16x16 feature maps).
      2. Detector Density only (classical white top-hat + DBSCAN clusters).
      3. Product Map: CAM * (epsilon + (1-epsilon)*Detector_Density) explicitly labelled as "Combined Localization".
  - Evaluates on the held-out annotated calcification test cohort (N=137).
  - Tests Pre-Registered Criterion 1: Product map must achieve >= +5.0% peak-in-lesion hit rate
    without increasing 3x3 TV distance by > 0.05.
  - Computes Agreement Score (CAM energy within detector cluster bounding boxes).
  - Generates:
      * paper/figures/p3b_localization_sharpening_comparison.png
      * paper/tables/localization_sharpening_comparison.json
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

_repo_dir = Path(__file__).resolve().parent.parent
if str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
Path("/tmp/matplotlib_cache").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataset import get_train_val_test_splits
from src.data.preprocess import read_image_array
from src.explain.detector import MicrocalcificationDetector
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


def compute_map_metrics(heatmap: np.ndarray, annot, grid) -> Tuple[bool, float, Tuple[int, int]]:
    """Compute peak hit, TV distance, and peak coordinate for any 512x512 heatmap."""
    py, px = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    hit = check_peak_inside_lesion((py, px), annot, margin=15.0)

    # Compute heat fraction per 3x3 cell
    cell_fractions = {}
    total_heat = float(np.sum(heatmap))
    if total_heat > 0:
        for c in grid.cells:
            y0, x0, y1, x1 = c.bbox
            cell_heat = float(np.sum(heatmap[y0:y1, x0:x1]))
            cell_fractions[c.cell_name] = cell_heat / total_heat
    else:
        for c in grid.cells:
            cell_fractions[c.cell_name] = 1.0 / 9.0

    annot_dist, _ = compute_annotation_cell_distribution(annot, grid)
    tv_dist = compute_total_variation_distance(cell_fractions, annot_dist)

    return hit, tv_dist, (py, px)


def compute_agreement_score(cam_heatmap: np.ndarray, clusters: list) -> float:
    """Compute fraction of CAM energy residing inside detected cluster bounding boxes."""
    if not clusters:
        return 0.0
    h, w = cam_heatmap.shape
    cluster_mask = np.zeros((h, w), dtype=bool)
    for c in clusters:
        y0, x0, y1, x1 = c.bbox_512
        iy0 = max(0, int(round(y0)))
        iy1 = min(h, int(round(y1)))
        ix0 = max(0, int(round(x0)))
        ix1 = min(w, int(round(x1)))
        # Pad cluster bbox by 10 px
        py0 = max(0, iy0 - 10)
        py1 = min(h, iy1 + 10)
        px0 = max(0, ix0 - 10)
        px1 = min(w, ix1 + 10)
        cluster_mask[py0:py1, px0:px1] = True

    total_energy = float(np.sum(cam_heatmap))
    if total_energy <= 0.0:
        return 0.0
    in_box_energy = float(np.sum(cam_heatmap[cluster_mask]))
    return float(in_box_energy / total_energy)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    # 1. Load Data
    manifest_path = Path("data/manifest.csv")
    annot_path = Path("data/annotations.csv")
    mdf = pd.read_csv(manifest_path)
    annot_df = pd.read_csv(annot_path)

    # Load held-out test split (seed 42)
    _, _, test_df = get_train_val_test_splits(mdf, seed=42, val_size=0.20)

    # Filter to annotated calcification test cases
    annot_calcs = annot_df[(annot_df["split"] == "test") & (annot_df["finding_type"] == "calcification")]
    calc_test_img_ids = set(annot_calcs["image_id"].astype(str).unique())
    test_calc_df = test_df[test_df["image_id"].astype(str).isin(calc_test_img_ids)].copy()

    logger.info("Evaluating on Annotated Calcification Test Cohort: %d images", len(test_calc_df))

    # 2. Load Model & Detector
    ckpt_path = Path("results/runs/20261103_222946-d91a0e4a-e899c89/ckpt/best.pt")
    model = build_model(num_classes=3, pretrained=False, bn_mode="trainable")
    ckpt_mgr = CheckpointManager(ckpt_path.parent)
    ckpt_mgr.load(ckpt_path, model=model, device=device)
    model.to(device)
    model.eval()

    gradcam = GradCAM(model)
    detector = MicrocalcificationDetector(
        kernel_size=7,
        std_multiplier=3.2,
        min_speck_area=2,
        max_speck_area=150,
        cluster_eps_native=80.0,
        cluster_min_specks=3,
        max_clusters_per_image=16,
        patch_size=256,
    )

    hits_cam = []
    hits_det = []
    hits_prod = []

    tv_cam = []
    tv_det = []
    tv_prod = []

    agreement_scores = []
    gallery_cases = []

    for idx, (_, row) in enumerate(test_calc_df.iterrows()):
        npy_path = Path("data") / row["cached_path"]
        if not npy_path.exists():
            continue

        arr_512 = np.load(npy_path)
        img_t = torch.from_numpy(arr_512).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

        grid = fit_anatomical_grid(arr_512, view=str(row["view"]))
        annot = parse_image_annotations(annot_df, row["image_id"])
        if annot is None:
            continue

        # 1. Compute Pure Grad-CAM
        loc_res = gradcam.analyze_localization(img_t, grid, class_idx=2)
        cam_map = loc_res.heatmap
        hit_c, tv_c, p_c = compute_map_metrics(cam_map, annot, grid)

        # 2. Run High-Resolution Detector
        arr_raw, _ = read_image_array(row["path"])
        ymin, xmin, ymax, xmax = [int(v) for v in row["crop_box"].split(",")]
        cropped = arr_raw[ymin : ymax + 1, xmin : xmax + 1]
        if row["flip"]:
            cropped = np.fliplr(cropped)
        scale = float(row["scale"])
        tissue_mask = cropped > 0.05

        specks, _ = detector.detect_specks(cropped, tissue_mask)
        clusters = detector.cluster_specks(specks, cropped, scale=scale, grid=grid, extract_patches=False)

        det_map = detector.compute_detector_density_map(clusters, target_size=(512, 512), sigma=12.0)
        hit_d, tv_d, p_d = compute_map_metrics(det_map, annot, grid)

        # 3. Product Map: CAM * (epsilon + (1 - epsilon)*Detector_Density)
        prod_map = detector.combine_cam_and_detector(cam_map, det_map, epsilon=0.20)
        hit_p, tv_p, p_p = compute_map_metrics(prod_map, annot, grid)

        # 4. Agreement Score
        agr = compute_agreement_score(cam_map, clusters)
        agreement_scores.append(agr)

        hits_cam.append(hit_c)
        hits_det.append(hit_d)
        hits_prod.append(hit_p)

        tv_cam.append(tv_c)
        tv_det.append(tv_d)
        tv_prod.append(tv_p)

        # Collect gallery candidates
        if hit_p and not hit_c and len(gallery_cases) < 4:
            gallery_cases.append({
                "site": row["site"],
                "view": f"{row['laterality']} {row['view']}",
                "label": int(row["label"]),
                "img_arr": arr_512,
                "grid": grid,
                "annot": annot,
                "cam_map": cam_map,
                "det_map": det_map,
                "prod_map": prod_map,
                "p_c": p_c,
                "p_p": p_p,
                "agr": agr,
            })

    # Quantitative summary
    n_cases = len(hits_cam)
    rate_cam = float(np.mean(hits_cam))
    rate_det = float(np.mean(hits_det))
    rate_prod = float(np.mean(hits_prod))

    mean_tv_cam = float(np.mean(tv_cam))
    mean_tv_det = float(np.mean(tv_det))
    mean_tv_prod = float(np.mean(tv_prod))
    mean_agr = float(np.mean(agreement_scores))

    delta_hit = rate_prod - rate_cam
    delta_tv = mean_tv_prod - mean_tv_cam
    criterion_met = (delta_hit >= 0.05) and (delta_tv <= 0.05)

    logger.info("\n=== Localization Sharpening Evaluation (N=%d Calcification Cases) ===", n_cases)
    logger.info("1. CAM Only:       Hit Rate = %.2f%% | Mean TV Distance = %.3f", rate_cam * 100.0, mean_tv_cam)
    logger.info("2. Detector Only:  Hit Rate = %.2f%% | Mean TV Distance = %.3f", rate_det * 100.0, mean_tv_det)
    logger.info("3. Product Map:    Hit Rate = %.2f%% | Mean TV Distance = %.3f (Delta Hit = %+.2f%%, Delta TV = %+.3f)",
                rate_prod * 100.0, mean_tv_prod, delta_hit * 100.0, delta_tv)
    logger.info("4. Agreement Score: Mean CAM Energy in Detector BBoxes = %.2f%%", mean_agr * 100.0)
    logger.info("Pre-Registered Criterion 1 Status (>= +5.0%% Hit Rate Gain): %s", "PASSED" if criterion_met else "FAILED / NULL")

    summary_table = {
        "cohort": f"Calcification Test Split (N={n_cases} annotated cases)",
        "cam_only": {"hit_rate": rate_cam, "mean_tv_distance": mean_tv_cam},
        "detector_only": {"hit_rate": rate_det, "mean_tv_distance": mean_tv_det},
        "product_map_combined": {
            "hit_rate": rate_prod,
            "mean_tv_distance": mean_tv_prod,
            "delta_hit_rate": delta_hit,
            "delta_tv_distance": delta_tv,
        },
        "mean_agreement_score": mean_agr,
        "pre_registered_criterion_1": {
            "target_delta_hit_rate": 0.05,
            "measured_delta_hit_rate": delta_hit,
            "target_max_delta_tv": 0.05,
            "measured_delta_tv": delta_tv,
            "status": "PASSED" if criterion_met else "FAILED / NULL",
        },
    }

    out_table_path = Path("paper/tables/localization_sharpening_comparison.json")
    out_table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_table_path, "w") as fp:
        json.dump(summary_table, fp, indent=2)
    logger.info("Saved sharpening quantitative table to %s", out_table_path)

    # Plot visual comparison figure
    if gallery_cases:
        fig, axes = plt.subplots(len(gallery_cases), 4, figsize=(16, 4 * len(gallery_cases)), dpi=300)
        if len(gallery_cases) == 1:
            axes = np.expand_dims(axes, 0)

        for row_idx, case in enumerate(gallery_cases):
            arr = case["img_arr"]
            annot = case["annot"]
            grid = case["grid"]

            # Col 0: Input Image + GT BBoxes + 3x3 Grid
            axes[row_idx, 0].imshow(arr, cmap="gray")
            for cell in grid.cells:
                y0, x0, y1, x1 = cell.bbox
                axes[row_idx, 0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="cyan", linewidth=1.0, linestyle=":")
            for x0, y0, x1, y1 in annot.boxes:
                axes[row_idx, 0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="lime", linewidth=2.0)
            axes[row_idx, 0].set_title(f"{case['site'].upper()} ({case['view']})\nInput & GT Annotation", fontsize=10, fontweight="bold")
            axes[row_idx, 0].axis("off")

            # Col 1: CAM Only (Coarse)
            axes[row_idx, 1].imshow(arr, cmap="gray")
            axes[row_idx, 1].imshow(case["cam_map"], cmap="jet", alpha=0.45)
            py_c, px_c = case["p_c"]
            axes[row_idx, 1].scatter([px_c], [py_c], color="red", marker="x", s=90, linewidths=2.5)
            axes[row_idx, 1].set_title("1. Pure Grad-CAM (Coarse)\nPeak Missed Lesion", fontsize=10, fontweight="bold")
            axes[row_idx, 1].axis("off")

            # Col 2: Detector Density Map
            axes[row_idx, 2].imshow(arr, cmap="gray")
            axes[row_idx, 2].imshow(case["det_map"], cmap="hot", alpha=0.55)
            axes[row_idx, 2].set_title("2. Detector Density\n(Speck Clusters)", fontsize=10, fontweight="bold")
            axes[row_idx, 2].axis("off")

            # Col 3: Product Map (Combined Localization)
            axes[row_idx, 3].imshow(arr, cmap="gray")
            axes[row_idx, 3].imshow(case["prod_map"], cmap="jet", alpha=0.45)
            py_p, px_p = case["p_p"]
            axes[row_idx, 3].scatter([px_p], [py_p], color="lime", marker="x", s=110, linewidths=3.0)
            axes[row_idx, 3].set_title("3. Combined Product (CAM x Det)\nSharpened Peak Hit GT Lesion", fontsize=10, fontweight="bold")
            axes[row_idx, 3].axis("off")

        plt.suptitle("Localization Sharpening on Microcalcifications: Pure CAM vs. Detector vs. Combined Product",
                     fontsize=13, fontweight="bold", y=0.99)
        fig.tight_layout()
        fig_out_path = Path("paper/figures/p3b_localization_sharpening_comparison.png")
        fig.savefig(fig_out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved sharpening visual comparison gallery to %s", fig_out_path)


if __name__ == "__main__":
    main()
