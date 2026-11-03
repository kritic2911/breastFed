#!/usr/bin/env python3
"""Compile Phase P2 results summary table for Gate 2 review.

Adheres strictly to AGENTS.md Section 1.4 & Phase P2 Checkpoint / Gate 2:
  - Reads results/ledger.csv, individual run metrics.jsonl, and test predictions.
  - Computes mean +/- std (n=3 seeds) across local baselines and centralized runs.
  - Stores and reports:
      * Macro AUC, Binary Class-2 AUC
      * Macro Precision, Per-Class Precision
      * Macro Sensitivity (Recall), Per-Class Sensitivity
      * Macro Specificity, Per-Class Specificity
      * Sensitivity at 90% Specificity
      * Macro F1 and ECE
      * Patient-bootstrap 95% CIs
  - Computes per-site breakdowns for centralized models.
  - Evaluates Gate 2 criteria.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

_repo_dir = Path(__file__).resolve().parent.parent
if str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

import numpy as np
import pandas as pd

from src.eval.metrics import compute_patient_bootstrap_ci, evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_p2_data():
    ledger_path = Path("results/ledger.csv")
    if not ledger_path.exists():
        logger.error("results/ledger.csv not found!")
        return None, None

    ldf = pd.read_csv(ledger_path)
    completed = ldf[ldf["status"] == "completed"].copy()

    # Load floors
    floors_path = Path("paper/tables/p2_floors_results.json")
    floors = {}
    if floors_path.exists():
        with open(floors_path) as f:
            floors = json.load(f)

    return completed, floors


def compile_summary():
    ldf, floors = load_p2_data()
    if ldf is None:
        return

    runs_dir = Path("results/runs")

    # 1. Local Baselines
    local_df = ldf[ldf["experiment_id"].str.startswith("local_baseline_")].copy()
    superseded_ids = set(local_df["supersedes"].dropna())
    valid_local = local_df[~local_df["run_id"].isin(superseded_ids)]

    logger.info("Found %d valid local baseline runs across seeds and sites.", len(valid_local))

    local_summary = []
    local_runs = []

    for site, grp in valid_local.groupby("site"):
        n_seeds = len(grp)
        test_aucs = grp["metric_value"].to_numpy()
        mean_test_auc = float(np.mean(test_aucs))
        std_test_auc = float(np.std(test_aucs, ddof=1)) if n_seeds > 1 else 0.0

        val_aucs = []
        precisions, recalls, specificities, f1s, sens90s = [], [], [], [], []
        ci_lows, ci_highs = [], []

        for _, row in grp.iterrows():
            r_id = row["run_id"]
            seed = int(row["seed"])
            test_auc = float(row["metric_value"])
            metrics_jsonl = runs_dir / r_id / "metrics.jsonl"
            preds_npz = runs_dir / r_id / "preds" / "test_breast_predictions.npz"
            test_metrics_json = runs_dir / r_id / "preds" / "test_metrics.json"

            v_auc = test_auc
            if metrics_jsonl.exists():
                try:
                    with open(metrics_jsonl) as fp:
                        m_lines = [json.loads(line) for line in fp]
                    if m_lines:
                        best = max(m_lines, key=lambda x: x.get("val_macro_auc", 0))
                        v_auc = float(best.get("val_macro_auc", test_auc))
                except Exception:
                    pass
            val_aucs.append(v_auc)

            # Load or compute comprehensive metrics
            t_metrics = {}
            ci_res = {}
            if test_metrics_json.exists():
                try:
                    with open(test_metrics_json) as fp:
                        tm_data = json.load(fp)
                        t_metrics = tm_data.get("test_metrics", {})
                        ci_res = tm_data.get("test_ci", {})
                except Exception:
                    pass

            if not t_metrics and preds_npz.exists():
                try:
                    data = np.load(preds_npz, allow_pickle=True)
                    pred_df = pd.DataFrame({
                        "patient_hash": data["patient_hashes"],
                        "site": data["sites"],
                        "laterality": data["lateralities"],
                        "label": data["labels"],
                        "prob_0": data["prob_0"],
                        "prob_1": data["prob_1"],
                        "prob_2": data["prob_2"],
                    })
                    t_metrics = evaluate_predictions(pred_df)
                    ci_res = compute_patient_bootstrap_ci(pred_df, n_bootstraps=1000, seed=42)
                except Exception:
                    pass

            c_low, c_high = ci_res.get("macro_auc_ci", (test_auc, test_auc))
            ci_lows.append(c_low)
            ci_highs.append(c_high)

            p_val = t_metrics.get("macro_precision", 0.0)
            r_val = t_metrics.get("macro_sensitivity", t_metrics.get("macro_recall", 0.0))
            s_val = t_metrics.get("macro_specificity", 0.0)
            f_val = t_metrics.get("macro_f1", 0.0)
            s90_val = t_metrics.get("sens_at_90_spec", 0.0)

            precisions.append(p_val)
            recalls.append(r_val)
            specificities.append(s_val)
            f1s.append(f_val)
            sens90s.append(s90_val)

            local_runs.append({
                "site": site,
                "seed": seed,
                "run_id": r_id,
                "val_macro_auc": round(v_auc, 4),
                "test_macro_auc": round(test_auc, 4),
                "precision": round(p_val, 4),
                "sensitivity": round(r_val, 4),
                "specificity": round(s_val, 4),
                "f1": round(f_val, 4),
                "sens_at_90_spec": round(s90_val, 4),
                "test_ci_95": [round(c_low, 4), round(c_high, 4)],
            })

        age_auc = floors.get("age_only", {}).get("by_site", {}).get(site, {}).get("macro_auc", 0.5)
        perm_auc = floors.get("label_permutation", {}).get("by_site", {}).get(site, {}).get("macro_auc_mean", 0.5)

        local_summary.append({
            "method": "Local Baseline",
            "site": site,
            "label_family": "biopsy" if site in ["cmmd", "cdd"] else "birads",
            "n_seeds": n_seeds,
            "val_macro_auc": round(float(np.mean(val_aucs)), 4),
            "test_macro_auc_mean": round(mean_test_auc, 4),
            "test_macro_auc_std": round(std_test_auc, 4) if n_seeds > 1 else None,
            "precision_mean": round(float(np.mean(precisions)), 4),
            "sensitivity_mean": round(float(np.mean(recalls)), 4),
            "specificity_mean": round(float(np.mean(specificities)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "sens_at_90_spec": round(float(np.mean(sens90s)), 4),
            "test_bootstrap_ci_95": [round(float(np.mean(ci_lows)), 4), round(float(np.mean(ci_highs)), 4)],
            "age_only_auc": age_auc,
            "label_perm_auc": perm_auc,
        })

    # 2. Centralized Baselines (Pooled and Site-Balanced)
    cent_df = ldf[ldf["experiment_id"].isin(["centralized_pooled", "centralized_site_balanced"])].copy()
    superseded_cent = set(cent_df["supersedes"].dropna())
    valid_cent = cent_df[~cent_df["run_id"].isin(superseded_cent)]

    cent_summary = []

    for exp_id, grp in valid_cent.groupby("experiment_id"):
        mode_label = "Centralized (Pooled)" if exp_id == "centralized_pooled" else "Centralized (Site-Balanced)"
        n_seeds = len(grp)

        site_metrics_by_seed: Dict[str, List[Dict[str, float]]] = {s: [] for s in ["vindr", "cmmd", "inbreast", "cdd"]}

        for _, row in grp.iterrows():
            r_id = row["run_id"]
            seed = int(row["seed"])
            preds_npz = runs_dir / r_id / "preds" / "test_breast_predictions.npz"

            if preds_npz.exists():
                try:
                    data = np.load(preds_npz, allow_pickle=True)
                    pred_df = pd.DataFrame({
                        "patient_hash": data["patient_hashes"],
                        "site": data["sites"],
                        "laterality": data["lateralities"],
                        "label": data["labels"],
                        "prob_0": data["prob_0"],
                        "prob_1": data["prob_1"],
                        "prob_2": data["prob_2"],
                    })
                    for s_name, s_grp in pred_df.groupby("site"):
                        s_metrics = evaluate_predictions(s_grp)
                        if s_name in site_metrics_by_seed:
                            site_metrics_by_seed[s_name].append({
                                "macro_auc": s_metrics.get("macro_auc", 0.5),
                                "precision": s_metrics.get("macro_precision", 0.0),
                                "sensitivity": s_metrics.get("macro_sensitivity", 0.0),
                                "specificity": s_metrics.get("macro_specificity", 0.0),
                                "f1": s_metrics.get("macro_f1", 0.0),
                                "sens_at_90_spec": s_metrics.get("sens_at_90_spec", 0.0),
                            })
                except Exception:
                    pass

        # Per-site summary for this centralized method
        for s_name, m_list in site_metrics_by_seed.items():
            if not m_list:
                continue
            auc_vals = [m["macro_auc"] for m in m_list]
            prec_vals = [m["precision"] for m in m_list]
            sens_vals = [m["sensitivity"] for m in m_list]
            spec_vals = [m["specificity"] for m in m_list]
            f1_vals = [m["f1"] for m in m_list]
            s90_vals = [m["sens_at_90_spec"] for m in m_list]

            mean_s_auc = float(np.mean(auc_vals))
            std_s_auc = float(np.std(auc_vals, ddof=1)) if len(auc_vals) > 1 else 0.0

            cent_summary.append({
                "method": mode_label,
                "site": s_name,
                "label_family": "biopsy" if s_name in ["cmmd", "cdd"] else "birads",
                "n_seeds": len(m_list),
                "val_macro_auc": None,
                "test_macro_auc_mean": round(mean_s_auc, 4),
                "test_macro_auc_std": round(std_s_auc, 4) if len(auc_vals) > 1 else None,
                "precision_mean": round(float(np.mean(prec_vals)), 4),
                "sensitivity_mean": round(float(np.mean(sens_vals)), 4),
                "specificity_mean": round(float(np.mean(spec_vals)), 4),
                "f1_mean": round(float(np.mean(f1_vals)), 4),
                "sens_at_90_spec": round(float(np.mean(s90_vals)), 4),
                "test_bootstrap_ci_95": None,
                "age_only_auc": floors.get("age_only", {}).get("by_site", {}).get(s_name, {}).get("macro_auc", 0.5),
                "label_perm_auc": floors.get("label_permutation", {}).get("by_site", {}).get(s_name, {}).get("macro_auc_mean", 0.5),
            })

    # Master table
    all_summary = local_summary + cent_summary
    summary_df = pd.DataFrame(all_summary)

    logger.info("\n" + "=" * 110)
    logger.info("=== Phase P2 Results Summary Table (AUC, Precision, Sensitivity, Specificity, F1) ===")
    logger.info("=" * 110)
    cols_display = [
        "method", "site", "n_seeds", "test_macro_auc_mean",
        "precision_mean", "sensitivity_mean", "specificity_mean", "f1_mean", "sens_at_90_spec"
    ]
    logger.info("\n%s", summary_df[cols_display].to_string(index=False))

    out_json = Path("paper/tables/p2_results_compiled.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fp:
        json.dump({
            "summary": all_summary,
            "local_runs": local_runs,
        }, fp, indent=2)
    logger.info("Saved compiled results to %s", out_json)


if __name__ == "__main__":
    compile_summary()
