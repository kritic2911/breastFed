"""Evaluate Train/Val/Test AUC side-by-side, seed std, and Class-2 AUC across all P2 runs.

Adheres strictly to AGENTS.md Section 4 (Gate 2 Checkpoint):
  - Train/val/test AUC side by side; flag train-test gap and val/test inversion.
  - Mean +/- seed standard deviation (n=3) for all metrics.
  - Benign-vs-malignant (class-2-vs-rest) AUC per site and pooled.
  - Generates paper/tables/p2_gate2_comprehensive.json.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from src.data.dataset import (
    MaskedCrossEntropyLoss,
    create_dataloader,
    get_train_val_test_splits,
)
from src.eval.metrics import evaluate_predictions
from src.models.backbone import build_model
from src.train.local import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_std(values: List[float]) -> float:
    """Sample standard deviation (ddof=1)."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def compute_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def run_train_evaluation():
    manifest_path = Path("data/manifest.csv")
    ledger_path = Path("results/ledger.csv")
    runs_dir = Path("results/runs")

    if not manifest_path.exists() or not ledger_path.exists():
        logger.error("Manifest or ledger not found.")
        return

    mdf = pd.read_csv(manifest_path)
    ldf = pd.read_csv(ledger_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device for evaluation: %s", device)
    loss_fn = MaskedCrossEntropyLoss()

    # Filter valid completed runs (exclude superseded and sweeps/smoke)
    superseded = set(ldf["supersedes"].dropna())
    valid_df = ldf[~ldf["run_id"].isin(superseded) & (ldf["status"] == "completed")].copy()
    valid_df = valid_df[
        valid_df["experiment_id"].str.startswith("local_baseline_")
        | valid_df["experiment_id"].isin(["centralized_pooled", "centralized_site_balanced"])
    ].reset_index(drop=True)

    # Check if prior comprehensive json exists to reuse already-evaluated train metrics
    out_json = Path("paper/tables/p2_gate2_comprehensive.json")
    prior_runs_cache = {}
    if out_json.exists():
        try:
            with open(out_json) as fp:
                prior_data = json.load(fp)
                for pr in prior_data.get("runs", []):
                    if "run_id" in pr and "train_macro_auc" in pr and pr["train_macro_auc"] > 0:
                        prior_runs_cache[pr["run_id"]] = pr
        except Exception:
            pass

    # Track individual run records
    all_run_details = []

    # Cache for train/val data splits per (site/mode, seed)
    splits_cache: Dict[str, Any] = {}

    for idx, row in valid_df.iterrows():
        run_id = row["run_id"]
        exp_id = row["experiment_id"]
        method = row["method"]
        site = row["site"]
        seed = int(row["seed"])

        run_path = runs_dir / run_id
        ckpt_path = run_path / "ckpt" / "best.pt"
        metrics_jsonl = run_path / "metrics.jsonl"
        test_metrics_json = run_path / "preds" / "test_metrics.json"

        if not ckpt_path.exists() or not test_metrics_json.exists():
            continue

        logger.info("[%d/%d] Processing Run %s (%s, seed=%d)", idx + 1, len(valid_df), run_id, exp_id, seed)

        # 1. Load test metrics
        with open(test_metrics_json) as fp:
            test_m_data = json.load(fp)

        test_metrics = test_m_data.get("test_metrics", {})
        test_ci = test_m_data.get("test_ci", {})
        test_macro_auc = test_metrics.get("macro_auc", 0.0)
        test_class_2_auc = test_metrics.get("auc_class_2_vs_rest", 0.0)

        # 2. Load validation metrics from metrics.jsonl for the best epoch
        val_macro_auc = 0.0
        val_class_2_auc = 0.0
        best_epoch = 1
        if metrics_jsonl.exists():
            m_lines = [json.loads(line) for line in open(metrics_jsonl)]
            if m_lines:
                best_entry = max(m_lines, key=lambda x: x.get("val_macro_auc", 0.0))
                best_epoch = best_entry.get("epoch", 1)
                val_macro_auc = best_entry.get("val_macro_auc", 0.0)
                val_class_2_auc = best_entry.get("val_class_2_auc", 0.0)

        # 3. Train metrics (from cache if already computed, else evaluate)
        if run_id in prior_runs_cache:
            p_rec = prior_runs_cache[run_id]
            train_macro_auc = p_rec.get("train_macro_auc", 0.0)
            train_class_2_auc = p_rec.get("train_class_2_auc", 0.0)
            per_site_train = p_rec.get("per_site_train", {})
        else:
            split_key = f"{site}_{seed}"
            if split_key not in splits_cache:
                if site == "pooled":
                    tr_df, va_df, te_df = get_train_val_test_splits(mdf, seed=seed, val_size=0.20)
                else:
                    site_manifest = mdf[mdf["site"] == site].copy()
                    tr_df, va_df, te_df = get_train_val_test_splits(site_manifest, seed=seed, val_size=0.20)
                splits_cache[split_key] = (tr_df, va_df, te_df)
            else:
                tr_df, va_df, te_df = splits_cache[split_key]

            train_loader = create_dataloader(tr_df, data_dir="data", is_train=False, batch_size=32)

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = build_model(num_classes=3, pretrained=False, bn_mode="trainable")
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)

            _, train_breast_df, train_metrics = evaluate_model(model, train_loader, device, loss_fn)
            train_macro_auc = train_metrics.get("macro_auc", 0.0)
            train_class_2_auc = train_metrics.get("auc_class_2_vs_rest", 0.0)

            per_site_train = {}
            if site == "pooled":
                for s in ["vindr", "cmmd", "inbreast", "cdd"]:
                    s_tr_breast = train_breast_df[train_breast_df["site"] == s]
                    if not s_tr_breast.empty:
                        s_tr_metrics = evaluate_predictions(s_tr_breast)
                        per_site_train[s] = {
                            "macro_auc": s_tr_metrics.get("macro_auc", 0.0),
                            "class_2_auc": s_tr_metrics.get("auc_class_2_vs_rest", 0.0),
                        }

        # Calculate gaps and flags
        train_test_gap = train_macro_auc - test_macro_auc
        val_test_gap = val_macro_auc - test_macro_auc
        # Flag if train-test gap > 0.15 (potential overfitting) or val-test inversion |val - test| > 0.05
        is_val_test_inversion = abs(val_test_gap) > 0.05
        is_large_train_gap = train_test_gap > 0.15

        record = {
            "run_id": run_id,
            "experiment_id": exp_id,
            "method": method,
            "site": site,
            "seed": seed,
            "best_epoch": best_epoch,
            # AUCs side by side
            "train_macro_auc": round(train_macro_auc, 4),
            "val_macro_auc": round(val_macro_auc, 4),
            "test_macro_auc": round(test_macro_auc, 4),
            "train_test_gap": round(train_test_gap, 4),
            "val_test_gap": round(val_test_gap, 4),
            "is_large_train_gap": is_large_train_gap,
            "is_val_test_inversion": is_val_test_inversion,
            # Class-2 (benign-vs-malignant / suspicious-vs-rest) AUCs side by side
            "train_class_2_auc": round(train_class_2_auc, 4),
            "val_class_2_auc": round(val_class_2_auc, 4),
            "test_class_2_auc": round(test_class_2_auc, 4),
            # Detailed test metrics
            "test_precision": round(test_metrics.get("macro_precision", 0.0), 4),
            "test_recall": round(test_metrics.get("macro_recall", 0.0), 4),
            "test_sensitivity": round(test_metrics.get("macro_sensitivity", 0.0), 4),
            "test_specificity": round(test_metrics.get("macro_specificity", 0.0), 4),
            "test_f1": round(test_metrics.get("macro_f1", 0.0), 4),
            "test_sens_at_90_spec": round(test_metrics.get("sens_at_90_spec", 0.0), 4),
            "test_macro_auc_ci": test_ci.get("macro_auc_ci", [0.0, 0.0]),
            "test_class_2_auc_ci": test_ci.get("class_2_auc_ci", [0.0, 0.0]),
            "per_site_train": per_site_train,
            "per_site_test": {s: d.get("metrics", {}) for s, d in test_metrics.get("per_site", {}).items()},
        }
        all_run_details.append(record)

    # 4. Aggregate across seeds (n=3) by experiment
    df_runs = pd.DataFrame(all_run_details)
    summary_tables = []

    # Group by experiment/method and site
    for (exp_name, site_name), grp in df_runs.groupby(["experiment_id", "site"]):
        n_seeds = len(grp)

        summary_tables.append({
            "experiment_id": exp_name,
            "method": grp["method"].iloc[0],
            "site": site_name,
            "n_seeds": n_seeds,
            # Macro AUC: Train / Val / Test (Mean +/- Seed Std)
            "train_macro_auc_mean": round(compute_mean(grp["train_macro_auc"].tolist()), 4),
            "train_macro_auc_std": round(compute_std(grp["train_macro_auc"].tolist()), 4),
            "val_macro_auc_mean": round(compute_mean(grp["val_macro_auc"].tolist()), 4),
            "val_macro_auc_std": round(compute_std(grp["val_macro_auc"].tolist()), 4),
            "test_macro_auc_mean": round(compute_mean(grp["test_macro_auc"].tolist()), 4),
            "test_macro_auc_std": round(compute_std(grp["test_macro_auc"].tolist()), 4),
            # Gap and Flags
            "train_test_gap_mean": round(compute_mean(grp["train_test_gap"].tolist()), 4),
            "val_test_gap_mean": round(compute_mean(grp["val_test_gap"].tolist()), 4),
            # Class-2 (Benign vs Malignant) AUC (Mean +/- Seed Std)
            "train_class_2_auc_mean": round(compute_mean(grp["train_class_2_auc"].tolist()), 4),
            "train_class_2_auc_std": round(compute_std(grp["train_class_2_auc"].tolist()), 4),
            "val_class_2_auc_mean": round(compute_mean(grp["val_class_2_auc"].tolist()), 4),
            "val_class_2_auc_std": round(compute_std(grp["val_class_2_auc"].tolist()), 4),
            "test_class_2_auc_mean": round(compute_mean(grp["test_class_2_auc"].tolist()), 4),
            "test_class_2_auc_std": round(compute_std(grp["test_class_2_auc"].tolist()), 4),
            # Additional Metrics
            "test_precision_mean": round(compute_mean(grp["test_precision"].tolist()), 4),
            "test_precision_std": round(compute_std(grp["test_precision"].tolist()), 4),
            "test_sensitivity_mean": round(compute_mean(grp["test_sensitivity"].tolist()), 4),
            "test_sensitivity_std": round(compute_std(grp["test_sensitivity"].tolist()), 4),
            "test_specificity_mean": round(compute_mean(grp["test_specificity"].tolist()), 4),
            "test_specificity_std": round(compute_std(grp["test_specificity"].tolist()), 4),
            "test_f1_mean": round(compute_mean(grp["test_f1"].tolist()), 4),
            "test_f1_std": round(compute_std(grp["test_f1"].tolist()), 4),
            "test_sens_at_90_spec_mean": round(compute_mean(grp["test_sens_at_90_spec"].tolist()), 4),
            "test_sens_at_90_spec_std": round(compute_std(grp["test_sens_at_90_spec"].tolist()), 4),
        })

    # Also for Centralized runs, compile per-site breakdown across seeds
    centralized_per_site_summary = []
    for exp_id in ["centralized_pooled", "centralized_site_balanced"]:
        c_grp = df_runs[df_runs["experiment_id"] == exp_id]
        if c_grp.empty:
            continue

        for s in ["vindr", "cmmd", "inbreast", "cdd"]:
            tr_aucs = []
            te_aucs = []
            te_c2_aucs = []
            te_precs = []
            te_sens = []
            te_specs = []
            te_f1s = []

            for _, r in c_grp.iterrows():
                # Train per-site
                p_tr = r.get("per_site_train", {}).get(s, {})
                if p_tr:
                    tr_aucs.append(p_tr.get("macro_auc", 0.0))
                # Test per-site
                p_te = r.get("per_site_test", {}).get(s, {})
                if p_te:
                    te_aucs.append(p_te.get("macro_auc", 0.0))
                    te_c2_aucs.append(p_te.get("auc_class_2_vs_rest", p_te.get("class_2_auc", 0.0)))
                    te_precs.append(p_te.get("macro_precision", 0.0))
                    te_sens.append(p_te.get("macro_sensitivity", p_te.get("macro_recall", 0.0)))
                    te_specs.append(p_te.get("macro_specificity", 0.0))
                    te_f1s.append(p_te.get("macro_f1", 0.0))

            centralized_per_site_summary.append({
                "experiment_id": exp_id,
                "site": s,
                "n_seeds": len(te_aucs),
                "train_macro_auc_mean": round(compute_mean(tr_aucs), 4),
                "train_macro_auc_std": round(compute_std(tr_aucs), 4),
                "test_macro_auc_mean": round(compute_mean(te_aucs), 4),
                "test_macro_auc_std": round(compute_std(te_aucs), 4),
                "test_class_2_auc_mean": round(compute_mean(te_c2_aucs), 4),
                "test_class_2_auc_std": round(compute_std(te_c2_aucs), 4),
                "test_precision_mean": round(compute_mean(te_precs), 4),
                "test_precision_std": round(compute_std(te_precs), 4),
                "test_sensitivity_mean": round(compute_mean(te_sens), 4),
                "test_sensitivity_std": round(compute_std(te_sens), 4),
                "test_specificity_mean": round(compute_mean(te_specs), 4),
                "test_specificity_std": round(compute_std(te_specs), 4),
                "test_f1_mean": round(compute_mean(te_f1s), 4),
                "test_f1_std": round(compute_std(te_f1s), 4),
            })

    output_payload = {
        "summary": summary_tables,
        "centralized_per_site": centralized_per_site_summary,
        "runs": all_run_details,
    }

    out_json = Path("paper/tables/p2_gate2_comprehensive.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fp:
        json.dump(output_payload, fp, indent=2)

    logger.info("Saved Gate 2 comprehensive analysis to %s", out_json)


if __name__ == "__main__":
    run_train_evaluation()
