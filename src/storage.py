"""Centralized result storage, artifact persistence, and logging."""

import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch


def save_run_artifacts(
    method: str,
    train_site: str,
    seed: int,
    threshold: float,
    best_state: dict,
    history: list,
    own_results: dict,
    cross_results: dict,
    results_root: str = "results",
    extra_meta: dict = None,
    output_dir: Path = None,
) -> Path:
    """Save all artifacts for a run:
    - best.pt: model checkpoint weights
    - metrics.json: full metrics dictionary
    - training_history.csv: per-epoch loss, val AUC, sensitivity, VRAM, timing
    - predictions_{eval_site}.csv: per-sample test predictions
    - results/all_runs_summary.csv: appends standardized metrics for aggregate analysis
    - experiment_log.md: appends markdown summary
    """
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = root / method / train_site / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model checkpoint
    if best_state is not None:
        torch.save(best_state, out_dir / "best.pt")

    # 2. Per-epoch training history CSV
    if history:
        df_hist = pd.DataFrame(history)
        df_hist.to_csv(out_dir / "training_history.csv", index=False)

    # 3. Save test predictions CSVs per evaluation site
    for eval_site, res in cross_results.items():
        if "predictions" in res and res["predictions"]:
            df_preds = pd.DataFrame(res["predictions"])
            df_preds.to_csv(out_dir / f"predictions_{eval_site}.csv", index=False)
        # Also save standalone site metric file
        site_metrics_payload = {
            "eval_site": eval_site,
            "threshold": float(threshold),
            "metrics": {k: v for k, v in res.items() if k != "predictions"}
        }
        with open(out_dir / f"metrics_{eval_site}.json", "w") as sf:
            json.dump(site_metrics_payload, sf, indent=2, default=str)

    # 4. Save metrics.json (merging cross_site_test if already exists)
    merged_cross = dict(cross_results)
    if (out_dir / "metrics.json").exists():
        try:
            with open(out_dir / "metrics.json") as f:
                prev_data = json.load(f)
            if "cross_site_test" in prev_data and isinstance(prev_data["cross_site_test"], dict):
                prev_cross = prev_data["cross_site_test"]
                prev_cross.update(merged_cross)
                merged_cross = prev_cross
        except Exception:
            pass

    results_payload = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "trained_on": train_site,
        "seed": seed,
        "threshold": float(threshold),
        "own_site_test": own_results,
        "cross_site_test": merged_cross,
        "history": history,
    }
    if extra_meta:
        results_payload.update(extra_meta)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results_payload, f, indent=2, default=str)

    # 5. Append to central all_runs_summary.csv
    summary_csv_path = root / "all_runs_summary.csv"
    summary_rows = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    best_epoch = None
    best_val_auc = None
    if history:
        best_rec = max(history, key=lambda x: x.get("val_auc", -1))
        best_epoch = best_rec.get("epoch")
        best_val_auc = best_rec.get("val_auc")

    for eval_site, res in cross_results.items():
        img_m = res.get("image_level", {})
        pat_m = res.get("patient_level", {})
        ci = res.get("bootstrap_auc_ci", {})
        summary_rows.append({
            "timestamp": ts,
            "method": method,
            "train_site": train_site,
            "eval_site": eval_site,
            "is_own_site": (train_site == eval_site),
            "seed": seed,
            "auc": img_m.get("auc"),
            "auc_ci_lower": ci.get("lower"),
            "auc_ci_upper": ci.get("upper"),
            "sensitivity_at_spec90": img_m.get("sensitivity"),
            "specificity": img_m.get("specificity"),
            "patient_auc": pat_m.get("auc"),
            "patient_sensitivity": pat_m.get("sensitivity"),
            "threshold": threshold,
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
        })

    df_new = pd.DataFrame(summary_rows)
    if summary_csv_path.exists():
        df_new.to_csv(summary_csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(summary_csv_path, mode="w", header=True, index=False)

    # 6. Append to experiment_log.md
    log_file = Path("experiment_log.md")
    if log_file.exists():
        append_to_experiment_log(log_file, method, train_site, seed, cross_results, best_epoch, best_val_auc)

    print(f"  [Artifacts Saved] Checkpoint, logs, predictions -> {out_dir}")
    print(f"  [Summary Updated] Appended {len(summary_rows)} rows -> {summary_csv_path}")

    return out_dir


def append_to_experiment_log(log_file: Path, method: str, train_site: str, seed: int,
                             cross_results: dict, best_epoch: int, best_val_auc: float):
    """Format a clean markdown table entry in experiment_log.md."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"\n### Run: {method.upper()} | Train: `{train_site}` | Seed: `{seed}` ({ts})\n",
        f"- **Best Epoch**: {best_epoch} (Val AUC: {best_val_auc:.4f})" if best_epoch else "",
        "\n| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |",
        "|---|---|:---:|:---:|:---:|",
    ]
    for eval_site, res in cross_results.items():
        role = "Own site" if eval_site == train_site else "Cross-site"
        img_m = res.get("image_level", {})
        pat_m = res.get("patient_level", {})
        ci = res.get("bootstrap_auc_ci", {})
        auc_str = f"{img_m.get('auc', 0):.4f} [{ci.get('lower', 0):.4f}, {ci.get('upper', 0):.4f}]"
        sens_str = f"{img_m.get('sensitivity', 0):.4f}"
        pat_auc_str = f"{pat_m.get('auc', 0):.4f}" if 'auc' in pat_m else "N/A"
        lines.append(f"| **{eval_site}** | {role} | {auc_str} | {sens_str} | {pat_auc_str} |")
    lines.append("\n---\n")

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write("\n".join(filter(None, lines)))
