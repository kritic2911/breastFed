"""Evaluate microcalcification detector recall on INbreast point annotations and tune router thresholds.

Adheres strictly to AGENTS.md Phase P3b, Rule 9 Geometry, and Decision D-010:
  - Benchmark detector recall on INbreast point annotations on train/val splits (test untouched).
  - Evaluates cluster recall: fraction of radiologist calcification points falling within margin of a detected cluster.
  - Compares cluster count distributions between calcification images vs. non-calcification images to tune the router threshold.
  - Generates quantitative summary: paper/tables/detector_inbreast_recall.json.
"""

import json
import logging
import sys
from pathlib import Path

_repo_dir = Path(__file__).resolve().parent.parent
if str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.data.dataset import get_train_val_test_splits
from src.data.preprocess import read_image_array
from src.explain.detector import MicrocalcificationDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    mdf = pd.read_csv("data/manifest.csv")
    adf = pd.read_csv("data/annotations.csv")

    # Split dataset (seed 42)
    train_df, val_df, test_df = get_train_val_test_splits(mdf, seed=42, val_size=0.20)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    inbreast_df = train_val_df[train_val_df["site"] == "inbreast"].copy()
    logger.info("Evaluating on INbreast Train/Val cohort: %d images", len(inbreast_df))

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

    # Filter INbreast calcification annotations on train/val
    train_val_img_ids = set(inbreast_df["image_id"].astype(str))
    inb_calcs = adf[
        (adf["site"] == "inbreast")
        & (adf["finding_type"] == "calcification")
        & (adf["image_id"].astype(str).isin(train_val_img_ids))
    ]

    calc_images = set(inb_calcs["image_id"].astype(str).unique())
    logger.info("Found %d INbreast train/val images with calcifications (%d total GT points)",
                len(calc_images), len(inb_calcs))

    total_gt_points = 0
    hit_gt_points = 0
    margin_512 = 20.0  # ~1.7 mm margin in 512 space

    image_results = []

    for idx, (_, row) in enumerate(inbreast_df.iterrows()):
        img_id = str(row["image_id"])
        has_calc = img_id in calc_images

        # Load native cropped image
        arr, info = read_image_array(row["path"])
        ymin, xmin, ymax, xmax = [int(v) for v in row["crop_box"].split(",")]
        cropped = arr[ymin : ymax + 1, xmin : xmax + 1]
        if row["flip"]:
            cropped = np.fliplr(cropped)

        scale = float(row["scale"])
        tissue_mask = cropped > 0.05

        # Run detector
        specks, _ = detector.detect_specks(cropped, tissue_mask)
        clusters = detector.cluster_specks(specks, cropped, scale=scale, extract_patches=False)

        n_clusters = len(clusters)
        total_specks_in_clusters = sum(c.speck_count for c in clusters)
        max_speck_cluster = max([c.speck_count for c in clusters], default=0)

        img_hits = 0
        img_points = 0

        if has_calc:
            gt_sub = inb_calcs[inb_calcs["image_id"].astype(str) == img_id]
            img_points = len(gt_sub)
            total_gt_points += img_points

            gt_pts_512 = gt_sub[["cached_xmin", "cached_ymin"]].to_numpy()

            if clusters and len(gt_pts_512) > 0:
                cluster_centroids_512 = np.array([c.centroid_512 for c in clusters])  # (K, 2) (cx, cy)
                # Compute pairwise distance between each GT point and all cluster centroids
                for gx, gy in gt_pts_512:
                    dists = np.sqrt(np.sum((cluster_centroids_512 - np.array([gx, gy])) ** 2, axis=1))
                    if np.min(dists) <= margin_512:
                        img_hits += 1
                hit_gt_points += img_hits

        pt_recall = float(img_hits / max(1, img_points)) if img_points > 0 else 0.0

        image_results.append({
            "image_id": img_id,
            "has_calc": has_calc,
            "label": int(row["label"]),
            "n_specks_detected": len(specks),
            "n_clusters": n_clusters,
            "total_specks_in_clusters": total_specks_in_clusters,
            "max_speck_cluster": max_speck_cluster,
            "gt_points": img_points,
            "hit_points": img_hits,
            "point_recall": pt_recall,
        })

    res_df = pd.DataFrame(image_results)
    calc_res_df = res_df[res_df["has_calc"]]
    non_calc_res_df = res_df[~res_df["has_calc"]]

    overall_pt_recall = float(hit_gt_points / max(1, total_gt_points))
    calc_img_detection_rate = float((calc_res_df["n_clusters"] >= 1).mean())

    logger.info("\n=== INbreast Detector Recall Evaluation ===")
    logger.info("Total Ground-Truth Calcification Points Evaluated: %d", total_gt_points)
    logger.info("Points Hit by Detected Clusters (margin <= %.1f px in 512): %d (%.2f%% point recall)",
                margin_512, hit_gt_points, overall_pt_recall * 100.0)
    logger.info("Images with >= 1 Detected Cluster: Calc images = %.1f%% | Non-calc images = %.1f%%",
                calc_img_detection_rate * 100.0, (non_calc_res_df["n_clusters"] >= 1).mean() * 100.0)

    # Evaluate router thresholds: cluster count >= T
    logger.info("\n=== Router Threshold Search (Calcification vs. Non-Calcification) ===")
    best_t = 1
    best_f1 = 0.0
    router_sweeps = {}

    y_true = res_df["has_calc"].astype(int).to_numpy()
    for t in [1, 2, 3, 4, 5]:
        y_pred = (res_df["n_clusters"] >= t).astype(int).to_numpy()
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        router_sweeps[f"threshold_{t}_clusters"] = {
            "threshold": t,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
        logger.info("Threshold C >= %d clusters: Precision=%.4f | Recall=%.4f | F1=%.4f",
                    t, prec, rec, f1)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    summary = {
        "dataset": "INbreast (train/val split, n=326 images)",
        "total_gt_points": total_gt_points,
        "hit_gt_points": hit_gt_points,
        "overall_point_recall": overall_pt_recall,
        "calc_image_detection_rate": calc_img_detection_rate,
        "best_router_threshold_clusters": best_t,
        "best_router_f1": best_f1,
        "router_sweeps": router_sweeps,
    }

    out_json = Path("paper/tables/detector_inbreast_recall.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fp:
        json.dump(summary, fp, indent=2)
    logger.info("\nSaved INbreast detector evaluation to %s", out_json)


if __name__ == "__main__":
    main()

