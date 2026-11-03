#!/usr/bin/env python3
"""Offline Metrics Computation & Storage across ALL Runs.

Computes and permanently stores:
  - Macro Precision, Per-Class Precision
  - Macro Recall / Sensitivity, Per-Class Recall / Sensitivity
  - Macro Specificity, Per-Class Specificity
  - Sensitivity at 90% Specificity
  - Macro AUC, Binary Class-2 AUC, Macro F1, and ECE
  - 1,000 Patient Bootstrap Confidence Intervals

Saves:
  1. results/runs/<run_id>/preds/test_metrics.json for every run.
  2. paper/tables/master_metrics_summary.json (all runs x all metrics).
  3. paper/tables/p2_results_compiled.json.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_repo_dir = Path(__file__).resolve().parent.parent
if str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

from src.eval.metrics import compute_patient_bootstrap_ci, evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_all_runs():
    runs_dir = Path("results/runs")
    runs_dir = Path("results/runs")
    all_pred_files = sorted(runs_dir.glob("*/preds/test_breast_predictions.npz"))
    logger.info("Found %d completed run prediction files to process.", len(all_pred_files))

    detailed_run_records = []

    for pred_npz in all_pred_files:
        run_id = pred_npz.parent.parent.name
        run_dir = pred_npz.parent.parent
        cfg_file = run_dir / "config.yaml"

        exp_id = "unknown"
        site_name = "unknown"
        seed_val = 42
        method_name = "unknown"

        if cfg_file.exists():
            import yaml
            try:
                with open(cfg_file) as fp:
                    c = yaml.safe_load(fp) or {}
                    exp_id = c.get("experiment_id", exp_id)
                    site_name = c.get("site", site_name)
                    seed_val = int(c.get("seed", seed_val))
                    method_name = c.get("method", exp_id)
            except Exception:
                pass

        try:
            data = np.load(pred_npz, allow_pickle=True)
            pred_df = pd.DataFrame({
                "patient_hash": data["patient_hashes"],
                "site": data["sites"],
                "laterality": data["lateralities"],
                "label": data["labels"],
                "prob_0": data["prob_0"],
                "prob_1": data["prob_1"],
                "prob_2": data["prob_2"],
            })
        except Exception as e:
            logger.error("Failed to load %s: %s", pred_npz, e)
            continue

        # Evaluate comprehensive metrics
        metrics = evaluate_predictions(pred_df)
        test_metrics_json = pred_npz.parent / "test_metrics.json"

        ci_res = {}
        if test_metrics_json.exists() and test_metrics_json.stat().st_size > 0:
            try:
                with open(test_metrics_json) as fp:
                    old_data = json.load(fp)
                    ci_res = old_data.get("test_ci", {})
            except Exception:
                pass

        if not ci_res:
            ci_res = compute_patient_bootstrap_ci(pred_df, n_bootstraps=500, seed=seed_val)

        # If pooled dataset, also evaluate per-site breakdown
        per_site_breakdown = {}
        if pred_df["site"].nunique() > 1:
            for s_name, s_grp in pred_df.groupby("site"):
                s_metrics = evaluate_predictions(s_grp)
                per_site_breakdown[s_name] = {
                    "metrics": s_metrics,
                }
            metrics["per_site"] = per_site_breakdown

        # Save test_metrics.json in run's preds/ directory
        with open(test_metrics_json, "w") as fp:
            json.dump({
                "run_id": run_id,
                "experiment_id": exp_id,
                "method": method_name,
                "site": site_name,
                "seed": seed_val,
                "test_metrics": metrics,
                "test_ci": ci_res,
            }, fp, indent=2, default=str)

        detailed_run_records.append({
            "run_id": run_id,
            "experiment_id": exp_id,
            "method": method_name,
            "site": site_name,
            "seed": seed_val,
            "macro_auc": round(metrics.get("macro_auc", 0.0), 4),
            "macro_f1": round(metrics.get("macro_f1", 0.0), 4),
            "macro_precision": round(metrics.get("macro_precision", 0.0), 4),
            "macro_recall": round(metrics.get("macro_recall", 0.0), 4),
            "macro_sensitivity": round(metrics.get("macro_sensitivity", 0.0), 4),
            "macro_specificity": round(metrics.get("macro_specificity", 0.0), 4),
            "sens_at_90_spec": round(metrics.get("sens_at_90_spec", 0.0), 4),
            "class_2_auc": round(metrics.get("auc_class_2_vs_rest", 0.0), 4),
            "class_2_sensitivity": round(metrics.get("sensitivity_class_2", metrics.get("sensitivity_class_pos", 0.0)), 4),
            "class_2_specificity": round(metrics.get("specificity_class_2", metrics.get("specificity_class_pos", 0.0)), 4),
            "class_2_precision": round(metrics.get("precision_class_2", metrics.get("precision_class_pos", 0.0)), 4),
            "bootstrap_ci": {k: [round(v[0], 4), round(v[1], 4)] for k, v in ci_res.items()},
            "per_site_breakdown": per_site_breakdown,
        })

    logger.info("Successfully processed and saved test_metrics.json for %d runs.", len(detailed_run_records))

    # Save master summary JSON
    out_master = Path("paper/tables/master_metrics_summary.json")
    out_master.parent.mkdir(parents=True, exist_ok=True)
    with open(out_master, "w") as fp:
        json.dump(detailed_run_records, fp, indent=2)
    logger.info("Saved master metrics summary to %s", out_master)

    # Recompile P2 summary table
    from scripts.compile_p2_summary import compile_summary
    compile_summary()


if __name__ == "__main__":
    process_all_runs()

